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
        timeout: int = 1800,
        filter_think_tags: bool = True,
        max_tokens: int = 32768,
        temperature: float = 1.0,
    ):
        url_env = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
        temperature_env = os.getenv("VLLM_TEMPERATURE", "1.0")
        max_tokens_env = os.getenv("VLLM_MAX_TOKENS", "32768")

        self.temperature = float(temperature_env) if temperature_env else temperature
        self.max_tokens = int(max_tokens_env) if max_tokens_env else max_tokens 
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
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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

For each rubric,
- Mark "PRESENT" if the criterion is satisfied, or "NOT_PRESENT" if it is not. For example, given the response "Apples are red", the rubric "Mentions apples" is PRESENT, "Does not mention strawberries" is also PRESENT since the response doesn't mention strawberries and "Mentions oranges" is NOT_PRESENT. Also, "Avoids mentioning strawberries" is PRESENT because the response doesn't mention strawberries. However, "Avoids mentioning apples" is NOT_PRESENT because the response mentions apples.
- If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be NOT PRESENT. Only return PRESENT if all of the criteria are met.
- One important exception to the above bullet point is that if a rubric says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria. That is, there are no partial credit for any of the criteria.

Start your response with a valid JSON object that starts with "```json" and ends with "```".

The keys must be the numbers of the rubrics provided and the values must be either "PRESENT" or "NOT_PRESENT" based on your evaluation. Ensure the JSON is valid and contains no extra text or explanations.

Example response:
```json
{{
 "1": "PRESENT",
 "2": "NOT_PRESENT",
 "3": "PRESENT"
}}
```

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

    def _coerce_results(data: dict) -> Dict[int, bool]:
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

    def _validate_count(results: Dict[int, bool]) -> bool:
        if expected_count and len(results) != expected_count:
            print(f"[grader debug] parsed count mismatch: expected={expected_count}, got={len(results)}")
            print(resp_text)
            # 修改点1
            return False
        return True

    # Prefer fenced JSON block first.
    match = re.search(r"```json\s*(\{.*?\})\s*```", resp_text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            data = json.loads(match.group(1))
            results = _coerce_results(data)
            return results if _validate_count(results) else {}
        except Exception:
            print("[grader debug] failed to parse fenced json block")
            print(resp_text)
            pass

    # Fallback to any JSON-like object in the text.
    match = re.search(r"\{.*\}", resp_text, re.DOTALL)
    if not match:
        return {}

    cleaned = match.group(0).strip()
    cleaned = re.sub(r",\s*}", "}", cleaned)

    try:
        data = json.loads(cleaned)
        results = _coerce_results(data)
        # 修改点2
        return results if _validate_count(results) else {}
    except Exception:
        print("[grader debug] failed to parse json object fallback")
        print(resp_text)
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
    
    for i, item in enumerate(rubric_items):
        verifier = item.tags.get("verifier", "llm")

        if verifier == "rule":
            rule_indices.append(i)
        else:
            llm_indices.append(i)

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



        # # === 新增：将 Prompt 写入日志文件 ===
        # import random
        # import datetime
        
        # # 1. 生成唯一的 Tag 和时间，方便和后面的 Result 对应（大致时间上对应）
        # p_tag = random.randint(1000, 9999)
        # p_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # # 2. 组装内容：加上显眼的箭头，因为 Prompt 通常很长
        # log_content = (
        #     f"\n[Time:{p_time}][ID:{p_tag}] >>>>>>>>> Grader Prompt 输入 (Start) >>>>>>>>>\n"
        #     f"{prompt_text}\n"
        #     f"[ID:{p_tag}] <<<<<<<<< Grader Prompt 输入 (End) <<<<<<<<<\n\n"
        # )
        
        # # 3. 写入同一个文件 (使用 'a' 追加模式)
        # # 建议使用绝对路径，确保所有进程写到一起
        # log_file_path = "/tmp/grader_debug_log.txt" 
        
        # try:
        #     with open(log_file_path, "a", encoding="utf-8") as f:
        #         f.write(log_content)
        # except Exception as e:
        #     print(f"写入 Prompt 日志失败: {e}", flush=True)
        # # =======================================


        

        # Async call to grader
        sampler_response = await grader_model([{"role": "user", "content": prompt_text}])

        # import random
        # import os
        # import datetime

        # tag = random.randint(1000, 9999)
        # timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # # 定义日志文件路径 (通常放在 /tmp 下最简单，或者您指定的任何绝对路径)
        # log_file_path = "grader_debug_log.txt"

        # # 组装要写入的文本内容
        # log_content = (
        #     f"\n[Time:{timestamp}][ID:{tag}] ==================== Grader 结果 ====================\n"
        #     f"[ID:{tag}] 1. Text: {sampler_response.response_text}\n"
        #     f"[ID:{tag}] 2. Metadata: {sampler_response.response_metadata}\n"
        #     f"[ID:{tag}] 3. Messages: {str(sampler_response.actual_queried_message_list)}\n"
        #     f"[ID:{tag}] ==================================================================\n"
        # )

        # # === 核心：写入文件 (使用 'a' 追加模式) ===
        # try:
        #     # encoding='utf-8' 保证中文不乱码
        #     with open(log_file_path, "a", encoding="utf-8") as f:
        #         f.write(log_content)
            
        #     # (可选) 如果您还想在控制台保留一份，这行可以留着，不想看可以注释掉
        #     # print(f"已写入日志到: {log_file_path}", flush=True) 
            
        # except Exception as e:
        #     # 万一写文件失败（比如磁盘满了），打印个错误，千万别让训练崩了
        #     print(f"写入临时文件失败: {e}", flush=True)

        llm_results = _parse_presence_response(sampler_response.response_text, len(llm_items))
        


        for local_idx, global_idx in enumerate(llm_indices):
            grading_response_list[global_idx] = {"criteria_met": llm_results.get(local_idx + 1, False)}

    return calculate_score(rubric_items, grading_response_list)





