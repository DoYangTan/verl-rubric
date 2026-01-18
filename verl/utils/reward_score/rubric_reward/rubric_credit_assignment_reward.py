import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import aiohttp
from dotenv import load_dotenv

# --- 可选依赖导入 (用于Token归因) ---
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# --- 尝试导入 verl 验证函数，如果没有则定义空函数 ---
try:
    from verl.utils.reward_score.rule_fn import get_verification_function
except ImportError:
    def get_verification_function(name):
        return None

load_dotenv()

# ==========================================
# 数据结构定义
# ==========================================

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

@dataclass
class SamplerResponse:
    response_text: str
    response_metadata: dict
    actual_queried_message_list: List[Dict[str, str]]

# ==========================================
# 异步采样器 (AsyncVLLMSampler)
# ==========================================

class AsyncVLLMSampler:
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: int = 1800, filter_think_tags: bool = True):
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
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=1000, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=self.timeout_val), trust_env=True)
        return self._session

    def _filter_think_tags(self, text: str) -> str:
        if not isinstance(text, str): return ""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _get_next_url(self) -> str:
        if not self.base_urls: return "http://localhost:8000/v1"
        selected_url = min(self.virtual_loads, key=self.virtual_loads.get)
        self.virtual_loads[selected_url] += 1
        return selected_url

    async def __call__(self, message_list: List[Dict[str, str]]) -> SamplerResponse:
        # 注意：这里设置 max_tokens 为 2048 足够容纳 JSON 返回
        payload = {
            "model": self.model,
            "messages": message_list,
            "temperature": 0.1,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"}
        }

        trial = 0
        current_url = None
        session = await self._get_session()

        while trial < 3:
            try:
                current_url = self._get_next_url()
                request_url = f"{current_url.rstrip('/')}/chat/completions".replace("/v1/v1", "/v1")
                
                async with session.post(request_url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        raise ValueError(f"HTTP {response.status}: {await response.text()}")
                    
                    response_data = await response.json()
                    content = response_data["choices"][0]["message"]["content"] or "{}"
                    
                    if self.filter_think_tags:
                        content = self._filter_think_tags(content)
                    
                    if current_url in self.virtual_loads:
                        self.virtual_loads[current_url] = max(0, self.virtual_loads[current_url] - 1)

                    return SamplerResponse(
                        response_text=content,
                        response_metadata={"usage": response_data.get("usage", {})},
                        actual_queried_message_list=message_list,
                    )
            except Exception:
                if current_url in self.virtual_loads:
                    self.virtual_loads[current_url] = max(0, self.virtual_loads[current_url] - 1)
                trial += 1
                await asyncio.sleep(0.5 * trial)

        return SamplerResponse(response_text="{}", response_metadata={}, actual_queried_message_list=message_list)

# ==========================================
# Token 归因计算器 (TokenAttributor)
# ==========================================

class TokenAttributor:
    def __init__(self, model_path: str = "gpt2"):
        self.tokenizer = None
        self.model_path = model_path
    
    def load_tokenizer(self):
        # 懒加载 tokenizer 以节省资源
        if self.tokenizer is None and AutoTokenizer is not None:
            try:
                path = os.getenv("TOKENIZER_PATH", self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            except Exception as e:
                print(f"[Attributor Warning] Could not load tokenizer: {e}")
                pass
    
    def find_quote_span(self, full_text: str, quote: str) -> Tuple[int, int]:
        """寻找引用在全文中的字符起止位置"""
        if not quote: return -1, -1
        # 简单查找，生产环境可替换为 fuzzy matching
        start = full_text.find(quote)
        if start != -1:
            return start, start + len(quote)
        return -1, -1

    def get_token_rewards(self, response_text: str, attributions: List[Dict]) -> List[float]:
        """核心逻辑：将 Quote 映射为 Token Reward Mask"""
        if self.tokenizer is None:
            self.load_tokenizer()
        if self.tokenizer is None:
            return [] # 没有 tokenizer 无法计算

        try:
            # 必须开启 return_offsets_mapping 才能做字符到 Token 的映射
            encoding = self.tokenizer(response_text, return_offsets_mapping=True, add_special_tokens=False)
            offsets = encoding.offset_mapping
            input_ids = encoding.input_ids
        except Exception:
            # Tokenizer 出错时的 fallback
            return [0.0] * len(response_text.split())

        token_rewards = [0.0] * len(input_ids)

        for attr in attributions:
            # 跳过未满足条件或没有引用的项
            if not attr.get("criteria_met") or not attr.get("quote"):
                continue
            
            points = attr.get("points", 0.0)
            quote = attr["quote"]
            
            # 1. 找到字符位置
            char_start, char_end = self.find_quote_span(response_text, quote)
            if char_start == -1: continue
            
            # 2. 映射到 Token ID
            target_token_idx = -1
            
            # 遍历 offsets 找到包含结束字符的那个 token
            for idx, (tok_start, tok_end) in enumerate(offsets):
                # 逻辑：只要 Token 的范围覆盖了 Quote 的结束点
                if tok_start < char_end and tok_end >= char_end:
                    target_token_idx = idx
                    break
                elif tok_start < char_end <= tok_end:
                    target_token_idx = idx
                    break
            
            # 兜底：如果没找到精确位置，找最后一个还没超过 char_end 的 token
            if target_token_idx == -1:
                 for idx, (tok_start, tok_end) in enumerate(offsets):
                    if tok_end <= char_end:
                        target_token_idx = idx
            
            # 3. 赋值奖励
            if target_token_idx != -1:
                token_rewards[target_token_idx] += points

        return token_rewards

# 全局单例
_global_attributor = None
def get_global_attributor():
    global _global_attributor
    if _global_attributor is None:
        _global_attributor = TokenAttributor(model_path=os.getenv("VLLM_MODEL", "gpt2"))
    return _global_attributor

# ==========================================
# Prompt 构建与解析
# ==========================================

def _format_prompt_messages(prompt: List[Dict[str, str]]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in prompt)

def _build_batch_grader_prompt(prompt: List[Dict[str, str]], response: str, rubric_items: List[RubricItem]) -> str:
    prompt_str = _format_prompt_messages(prompt)
    rubrics_str = "\n".join(f"{idx + 1}. (points: {item.points}) {item.criterion}" for idx, item in enumerate(rubric_items))
    
    # -----------------------------------------------------------
    # 为了防止 Markdown 渲染错误，我将 Prompt 拆分拼接
    # 你可以放心地保留这些代码，功能是完整的
    # -----------------------------------------------------------
    
    # 这是一个 trick，避免连续的三个反引号破坏 Python 字符串结构
    json_block_start = "```" + "json"
    json_block_end = "```"
    
    instruction = (
        "You are an expert evaluator. Given a user prompt, a generated response, and a list of quality rubrics, evaluate the response against EACH rubric.\n\n"
        "For each rubric:\n"
        "1. Determine if the criteria is \"PRESENT\" or \"NOT_PRESENT\".\n"
        "2. If \"PRESENT\", **you MUST extract the exact substring (quote)** from the response that satisfies the criteria.\n"
        "3. If \"NOT_PRESENT\", the quote must be null.\n\n"
        "Return a valid JSON object. Keys are rubric numbers. Values are objects with \"status\" and \"quote\".\n\n"
        "Example:\n"
        f"{json_block_start}\n"
        "{\n"
        " \"1\": {\"status\": \"PRESENT\", \"quote\": \"Apples are red\"},\n"
        " \"2\": {\"status\": \"NOT_PRESENT\", \"quote\": null}\n"
        "}\n"
        f"{json_block_end}\n"
    )

    # 最终拼接 Prompt
    final_prompt = (
        f"{instruction}\n"
        f"<Prompt>\n{prompt_str}\n</Prompt>\n\n"
        f"<Response>\n{response}\n</Response>\n\n"
        f"<Rubrics>\n{rubrics_str}\n</Rubrics>"
    )
    
    return final_prompt

def _parse_presence_response(resp_text: str, expected_count: int) -> Dict[int, Dict]:
    if not isinstance(resp_text, str): return {}

    def _coerce_results(data: dict) -> Dict[int, Dict]:
        results = {}
        for key, val in data.items():
            try:
                idx = int(key)
                status = False
                quote = None
                # 兼容简写 {"1": "PRESENT"}
                if isinstance(val, str):
                    if val.strip().upper() == "PRESENT": status = True
                # 处理完整格式 {"1": {"status": "PRESENT", "quote": "..."}}
                elif isinstance(val, dict):
                    if val.get("status", "").strip().upper() == "PRESENT": status = True
                    quote = val.get("quote")
                results[idx] = {"status": status, "quote": quote}
            except (ValueError, TypeError):
                continue
        return results

    # 提取 JSON 块
    match = re.search(r"```json\s*(\{.*?\})\s*```", resp_text, re.DOTALL | re.IGNORECASE)
    if not match:
        match = re.search(r"\{.*\}", resp_text, re.DOTALL)
    
    if match:
        try:
            cleaned = re.sub(r",\s*}", "}", match.group(1)) # 修复 JSON 尾随逗号
            data = json.loads(cleaned)
            return _coerce_results(data)
        except:
            pass
    return {}

# ==========================================
# 主评分逻辑
# ==========================================

_global_grader = None
def get_global_grader():
    global _global_grader
    if _global_grader is None:
        _global_grader = AsyncVLLMSampler(filter_think_tags=True)
    return _global_grader

async def async_grade_single_example(prompt: List[Dict[str, str]], response: str, rubric_items: List[RubricItem], grader_model) -> Tuple[float, List[float]]:
    grading_details = [None] * len(rubric_items)
    
    # 区分 Rule 检查 和 LLM 检查
    rule_indices = [idx for idx, item in enumerate(rubric_items) if item.tags and item.tags.get("verifier") == "rule"]
    llm_indices = [idx for idx, item in enumerate(rubric_items) if idx not in rule_indices]

    # 1. 执行规则检查
    for idx in rule_indices:
        item = rubric_items[idx]
        verify_func = get_verification_function(item.tags.get("function"))
        # 规则检查暂时不返回 quote
        met = verify_func(response, item.tags.get("parameters") or {}) if verify_func else False
        grading_details[idx] = {"criteria_met": met, "points": item.points, "quote": None}

    # 2. 执行 LLM 检查 (带归因)
    if llm_indices:
        llm_items = [rubric_items[i] for i in llm_indices]
        prompt_text = _build_batch_grader_prompt(prompt, response, llm_items)
        
        # 调用 VLLM
        sampler_response = await grader_model([{"role": "user", "content": prompt_text}])
        llm_results = _parse_presence_response(sampler_response.response_text, len(llm_items))
        
        for local_idx, global_idx in enumerate(llm_indices):
            res = llm_results.get(local_idx + 1, {"status": False, "quote": None})
            grading_details[global_idx] = {
                "criteria_met": res["status"],
                "points": rubric_items[global_idx].points,
                "quote": res["quote"]
            }

    # 3. 计算总分 (Scalar)
    total_possible = sum(r.points for r in rubric_items if r.points > 0)
    achieved = sum(d["points"] for d in grading_details if d and d["criteria_met"])
    scalar_score = float(max(0.0, achieved / total_possible)) if total_possible > 0 else 0.0

    # 4. 计算 Token 归因 (Vector)
    attributor = get_global_attributor()
    valid_details = [d for d in grading_details if d is not None]
    token_rewards = attributor.get_token_rewards(response, valid_details)

    return scalar_score, token_rewards

# ==========================================
# 外部调用入口
# ==========================================

async def compute_score(solution_str: str, ground_truth: Any, prompt: Any = None, **kwargs) -> float:
    try:
        # 解析 Ground Truth 中的 Rubric
        rm_data = ground_truth
        if isinstance(rm_data, str):
            try: rm_data = json.loads(rm_data)
            except: pass
        
        if not isinstance(rm_data, dict): return 0.0

        rubrics = rm_data.get("rubrics", []) or rm_data.get("Rubric", [])
        if not rubrics: return 0.0
            
        rubric_items = [RubricItem.from_dict(r) for r in rubrics]
        input_prompt = prompt if prompt is not None else kwargs.get("prompt", [])
        
        # 获取分数和归因
        score, token_rewards = await async_grade_single_example(input_prompt, solution_str, rubric_items, get_global_grader())
        
        # 注意: 
        # 如果你的训练框架(verl)支持接收 dense rewards，请取消下面这行的注释并修改返回类型
        # return score, token_rewards 
        
        return float(score), token_rewards

    except Exception as e:
        print(f"[RuscaRL Error] compute_score failed: {e}")
        return 0.0