"""
Process Reward Model (PRM) scorer.
Refactored to use requests instead of openai library.
"""

import asyncio
import json
import logging
import re
import httpx
from typing import Any, Optional

logger = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\boxed\{([-+]?\d+)\}")
_SCORE_RE = re.compile(r"Score:\s*([-+]?\d+)(?!\.)", re.IGNORECASE)

_GREEN = "\033[32m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def _sanitize_text(text: str) -> str:
    """Replace XML-like tags that may trigger content filters."""
    text = re.sub(r"<tool_call>.*?</tool_call>", "[tool_call block]", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]*>", '', text)
    return text


def _build_prm_judge_prompt(response_text: str, instruction_text: str) -> list[dict]:
    """Construct the judge messages for PRM evaluation with short Chain-of-Thought reasoning."""
    system = (
        "You are a response quality judge. Evaluate the response quality and then output a final score.\n\n"
        "First, write a 1-2 sentence brief analysis of the response quality (accuracy, completeness, and context alignment).\n"
        "Then, output the final score exactly in this format on a new line:\n"
        "Score: 1 = The response answers the instruction correctly, OR is a natural conversational reply (e.g. 'hello', 'thanks').\n"
        "Score: 0 = The response is partially correct but incomplete, vague, or lacks details.\n"
        "Score: -1 = The response is wrong, hallucinated, unsafe, or ignores the instruction.\n\n"
        "Example output:\n"
        "Analysis: The response perfectly satisfies all file-writing tasks.\n"
        "Score: 1"
    )
    clean_instruction = _sanitize_text(instruction_text)
    
    # Sanitize first, then truncate to avoid splitting tags
    full_sanitized_response = _sanitize_text(response_text)
    if len(full_sanitized_response) > 1500:
        logger.warning("Response truncated for PRM scoring")
        
    clean_response = full_sanitized_response[:1500]
    user = (
        f"Instruction: {clean_instruction}\n\n"
        f"Response: {clean_response}\n\n"
        "Provide your Analysis and final Score:"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_prm_score(text: str) -> Optional[int]:
    # Si el modelo local inyecta etiquetas think o texto extra, buscamos la expresión regular
    matches = _SCORE_RE.findall(text)
    if matches:
        val = int(matches[-1])
        if val in (1, -1, 0):
            return val
    matches = _BOXED_RE.findall(text)
    if matches:
        val = int(matches[-1])
        if val in (1, -1, 0):
            return val
    # No valid score found via regex — don't guess from random numbers
    logger.warning("No valid score pattern found in response, returning None")
    return None


def _majority_vote(votes: list[Optional[int]]) -> Optional[float]:
    valid = [v for v in votes if v is not None]
    if not valid:
        return None
    positive = sum(1 for v in valid if v > 0)
    negative = sum(1 for v in valid if v < 0)
    neutral = len(valid) - positive - negative

    if positive > negative and positive > neutral:
        return 1.0
    elif negative > positive and negative > neutral:
        return -1.0
    elif neutral > positive and neutral > negative:
        return 0.0
    else:
        return None  # Tie — ambiguous


class PRMScorer:
    def __init__(
        self,
        prm_url: str,
        prm_model: str = "local-model",
        prm_m: int = 1,
        temperature: float = 0.5,
        max_new_tokens: int = 50,
    ):
        # Handle provider-specific URL construction
        base_url = prm_url.rstrip('/')
        if "ollama" in base_url.lower():
            self.prm_url = f"{base_url}/api/chat"
        elif "/v1" not in base_url:
            self.prm_url = f"{base_url}/v1/chat/completions"
        else:
            self.prm_url = f"{base_url}/chat/completions"
            
        self.prm_model = prm_model
        self.prm_m = prm_m
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    async def evaluate(
        self,
        response: str,
        instruction: str,
        session_id: str = "",
        turn_num: int = 0,
    ) -> dict:
        msgs = _build_prm_judge_prompt(response, instruction)

        results = await asyncio.gather(
            *[self._query_once(msgs, i) for i in range(self.prm_m)]
        )

        scores = [r[0] for r in results]
        final = _majority_vote(scores)

        representative = ""
        if final is not None and final != 0.0:
            for s, text in results:
                if s is not None and s == int(final):
                    representative = text
                    break

        votes_display = [s if s is not None else "fail" for s in scores]
        logger.info(
            f"{_CYAN}[PRMScorer] session={session_id} turn={turn_num} "
            f"model={self.prm_model} votes={votes_display} → score={final}{_RESET}"
        )
        return {"score": final, "votes": votes_display, "eval_text": representative}

    async def _query_once(
        self, messages: list[dict], vote_id: int
    ) -> tuple[Optional[int], str]:
        try:
            payload = {
                "model": self.prm_model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_new_tokens,
            }
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(self.prm_url, json=payload)
            response.raise_for_status()
            choices = response.json().get('choices', [])
            if not choices:
                raise ValueError("No choices in response")
            content = choices[0].get('message', {}).get('content', '')
            return _parse_prm_score(content), content
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError, ValueError) as e:
            logger.warning("[PRMScorer] query failed (vote %d): %s", vote_id, e)
            return None, ""
