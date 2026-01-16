import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import aiohttp
from dotenv import load_dotenv

try:
    from verl.utils.reward_score.rule_fn import get_verification_function
except ImportError:
    def get_verification_function(name):
        return None

load_dotenv()

@dataclass
class RubricItem:
    criterion: str
    points: float
    tags: Dict[str, Any]

    def __str__(self) -> str:
        return self.criterion

    @classmethod
    def from_dict(cls, d: dict) -> "RubricItem":
        tags_data = d.get("tags", {})
        if isinstance(tags_data, list):
            tags_dict = {}
            for tag in tags_data:
                if isinstance(tag, str) and ":" in tag:
                    key, value = tag.split(":", 1)
                    tags_dict[key] = value
                elif isinstance(tag, str):
                    tags_dict[tag] = True
            tags_data = tags_dict
        elif not isinstance(tags_data, dict):
            tags_data = {}
        return cls(criterion=d["criterion"], points=d["points"], tags=tags_data)

    def to_dict(self) -> dict:
        return {"criterion": self.criterion, "points": self.points, "tags": self.tags}

@dataclass
class SamplerResponse:
    response_text: str
    response_metadata: dict
    actual_queried_message_list: List[Dict[str, str]]

class AsyncVLLMSampler:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        filter_think_tags: bool = True,
    ):
        url_env = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.base_urls = [base_url] if base_url else [url.strip() for url in url_env.split(',') if url.strip()]
        self.model = model or os.getenv("VLLM_MODEL", "default")
        self.virtual_loads = {url: 0 for url in self.base_urls}
        self.timeout_val = timeout
        self.filter_think_tags = filter_think_tags
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('VLLM_API_KEY', 'dummy')}" 
        }
        # Persistent session to avoid connection overhead
        self._session = None

    async def _get_session(self):
        """Get or create async session with high-performance pool"""
        if self._session is None or self._session.closed:
            # Increase limit for higher concurrency
            connector = aiohttp.TCPConnector(limit=1000, ttl_dns_cache=300)
            timeout = aiohttp.ClientTimeout(total=self.timeout_val)
            self._session = aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout,
                trust_env=True # Ensure no_proxy env var is handled
            )
        return self._session

    def _filter_think_tags(self, text: str) -> str:
        """Safely filter <think> tags with type check"""
        if not isinstance(text, str):
            return ""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _get_next_url(self) -> str:
        """Load balancing (Fill-the-gap)"""
        if not self.base_urls:
            return "http://localhost:8000/v1"
        selected_url = min(self.virtual_loads, key=self.virtual_loads.get)
        self.virtual_loads[selected_url] += 1
        return selected_url

    async def __call__(self, message_list: List[Dict[str, str]]) -> SamplerResponse:
        payload = {
            "model": self.model,
            "messages": message_list,
            "temperature": 0.2,
            "top_k": 20,
            "top_p": 0.8,
            "max_tokens": 32768
        }

        trial = 0
        current_url = None
        session = await self._get_session()

        while trial < 3:
            try:
                current_url = self._get_next_url()
                request_url = f"{current_url.rstrip('/')}/chat/completions"
                request_url = request_url.replace("/v1/v1", "/v1")

                async with session.post(request_url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"HTTP {response.status}: {error_text}")
                            
                    response_data = await response.json()
                    
                    if not response_data or "choices" not in response_data:
                         raise ValueError(f"Invalid response format: {response_data}")

                    content = response_data["choices"][0]["message"]["content"]
                    
                    # Handle None response to prevent crash
                    if content is None:
                        content = "{}"
                    
                    if self.filter_think_tags:
                        content = self._filter_think_tags(content)
                    
                    if current_url in self.virtual_loads:
                        self.virtual_loads[current_url] = max(0, self.virtual_loads[current_url] - 1)

                    return SamplerResponse(
                        response_text=content,
                        response_metadata={"usage": response_data.get("usage", {})},
                        actual_queried_message_list=message_list,
                    )

            except Exception as e:
                if current_url in self.virtual_loads:
                    self.virtual_loads[current_url] = max(0, self.virtual_loads[current_url] - 1)
                
                trial += 1
                await asyncio.sleep(0.5 * trial)

        return SamplerResponse(response_text="{}", response_metadata={}, actual_queried_message_list=message_list)

def _format_prompt_messages(prompt: List[Dict[str, str]]) -> str:
    """Convert prompt list to string"""
    return "\n".join(f"{m['role']}: {m['content']}" for m in prompt if m.get("role") != "system")

def _build_batch_grader_prompt(prompt: List[Dict[str, str]], response: str, rubric_items: List[RubricItem]) -> str:
    """Build prompt for LLM-as-a-Judge"""
    prompt_str = _format_prompt_messages(prompt)
    rubrics_str = "\n".join(f"{idx + 1}. (points: {item.points}) {item.criterion}" for idx, item in enumerate(rubric_items))
    
    return f'''You are an expert evaluator. Given a user prompt, a generated response, and a list of quality rubrics, please evaluate the response against EACH rubric.
Start your response with a valid JSON object that starts with "```json" and ends with "```".
The keys must be the numbers of the rubrics provided and the values must be either "PRESENT" or "NOT_PRESENT".

<Prompt>
{prompt_str}
</Prompt>

<Response>
{response}
</Response>

<Rubrics>
{rubrics_str}
</Rubrics>'''

def _parse_presence_response(resp_text: str, expected_count: int) -> Dict[int, bool]:
    """Parse JSON response with robust extraction"""
    if not isinstance(resp_text, str) or resp_text == "{}":
        return {}
        
    # Extract JSON block
    match = re.search(r'\{.*\}', resp_text, re.DOTALL)
    if not match:
        return {}
    
    cleaned = match.group(0).strip()
    # Fix common format errors
    cleaned = re.sub(r",\s*}", "}", cleaned)
    
    try:
        data = json.loads(cleaned)
        results = {}
        for key, val in data.items():
            try:
                idx = int(key)
                if isinstance(val, str):
                    norm = val.strip().upper()
                    if norm == "PRESENT":
                        results[idx] = True
                    elif norm == "NOT_PRESENT":
                        results[idx] = False
            except (ValueError, TypeError):
                continue
        return results
    except Exception:
        return {}

def calculate_score(rubric_items: List[RubricItem], grading_response_list: List[dict]) -> float:
    """Calculate normalized score [0, 1]"""
    total_possible_points = sum(r.points for r in rubric_items if r.points > 0)
    if total_possible_points == 0: 
        return 0.0
    
    achieved_points = 0
    for r, g in zip(rubric_items, grading_response_list):
        if g and g.get("criteria_met", False):
            achieved_points += r.points
            
    return float(max(0.0, achieved_points / total_possible_points))

_global_grader = None

def get_global_grader():
    global _global_grader
    if _global_grader is None:
        _global_grader = AsyncVLLMSampler(filter_think_tags=True)
    return _global_grader

async def async_grade_single_example(
    prompt: List[Dict[str, str]], 
    response: str,
    rubric_items: List[RubricItem],
    grader_model
) -> float:
    """Async scoring: Rule Check + Async LLM Grading"""
    grading_response_list = [None] * len(rubric_items)
    
    rule_indices = []
    llm_indices = []
    
    for idx, item in enumerate(rubric_items):
        if item.tags and item.tags.get("verifier") == "rule" and item.tags.get("function"):
            rule_indices.append(idx)
        else:
            llm_indices.append(idx)

    # Execute Rule Check (Sync)
    for idx in rule_indices:
        item = rubric_items[idx]
        func_name = item.tags.get("function")
        param = item.tags.get("parameters") or {}
        
        verify_func = get_verification_function(func_name)
        if verify_func:
            try:
                grading_response_list[idx] = {"criteria_met": verify_func(response, param)}
            except Exception:
                grading_response_list[idx] = {"criteria_met": False}
        else:
            grading_response_list[idx] = {"criteria_met": False}

    # Execute Async LLM Grading
    if llm_indices:
        llm_items = [rubric_items[i] for i in llm_indices]
        prompt_text = _build_batch_grader_prompt(prompt, response, llm_items)
        
        # Async call to grader
        sampler_response = await grader_model([{"role": "user", "content": prompt_text}])
        llm_results = _parse_presence_response(sampler_response.response_text, len(llm_items))
        
        for local_idx, global_idx in enumerate(llm_indices):
            grading_response_list[global_idx] = {"criteria_met": llm_results.get(local_idx + 1, False)}

    return calculate_score(rubric_items, grading_response_list)

async def compute_score(
    solution_str: str,
    ground_truth: Any,
    prompt: Any = None,
    extra_info: Dict[str, Any] = None,
    **kwargs
) -> float:
    """Entry point for verl RewardLoop"""
    try:
        rm_data = ground_truth

        if isinstance(rm_data, str):
            try:
                rm_data = json.loads(rm_data)
            except json.JSONDecodeError:
                pass
        
        if not isinstance(rm_data, dict):
            return 0.0

        rubrics = rm_data.get("rubrics", []) or rm_data.get("Rubric", [])
        
        if not rubrics:
            return 0.0
            
        rubric_items = [RubricItem.from_dict(r) for r in rubrics]
        grader = get_global_grader()
        
        # Ensure prompt is available (fetch from kwargs if not passed positionally)
        input_prompt = prompt if prompt is not None else kwargs.get("prompt", [])
        
        score = await async_grade_single_example(
            input_prompt, 
            solution_str, 
            rubric_items, 
            grader
        )
        return float(score)

    except Exception as e:
        print(f"[RuscaRL Error] compute_score failed: {e}")
        return 0.0