from typing import Optional
from dataclasses import dataclass
from openai import OpenAI
from config import OPENAI_MODEL, TOKEN_PRICING


@dataclass
class LLMResponse:
    """Container for LLM response with usage info"""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    def calculate_cost(self, model: str = None) -> float:
        """Calculate cost in USD based on token usage"""
        model = model or OPENAI_MODEL
        pricing = TOKEN_PRICING.get(model, TOKEN_PRICING["default"])
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


def call_llm(
    client: OpenAI,
    prompt: str,
    verbosity_level: str = "medium",
    reasoning_level: Optional[str] = None,
    return_usage: bool = False,
) -> str | LLMResponse:
    """
    Unified wrapper for OpenAI Responses API calls.
    
    Args:
        client: OpenAI client instance
        prompt: The input prompt/question
        verbosity_level: "low", "medium", or "high" - controls output verbosity
        reasoning_level: None, "low", "medium", or "high" - controls reasoning effort
        return_usage: If True, returns LLMResponse with token usage info
    
    Returns:
        The text output from the model, or LLMResponse if return_usage=True
    """
    if OPENAI_MODEL.startswith("gpt-4"):
        reasoning_level = None
        verbosity_level = "medium"
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        text={
            "verbosity": verbosity_level
        },
        reasoning={
            "effort": reasoning_level
        }
    )
    
    text = response.output_text.strip()
    
    if return_usage:
        usage = getattr(response, 'usage', None)
        if usage:
            return LLMResponse(
                text=text,
                input_tokens=getattr(usage, 'input_tokens', 0),
                output_tokens=getattr(usage, 'output_tokens', 0)
            )
        return LLMResponse(text=text)
    
    return text
