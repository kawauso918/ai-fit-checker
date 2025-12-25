"""
コスト追跡機能
LLM呼び出しのtoken数とコストを概算で表示
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CostInfo:
    """コスト情報"""
    provider: str  # "openai" or "anthropic"
    model_name: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# モデルごとの価格（2024年時点の概算、1K tokensあたり）
PRICING = {
    "openai": {
        "gpt-4o-mini": {
            "input": 0.15 / 1000,  # $0.15 per 1M tokens
            "output": 0.60 / 1000,  # $0.60 per 1M tokens
        },
        "gpt-4o": {
            "input": 2.50 / 1000,  # $2.50 per 1M tokens
            "output": 10.00 / 1000,  # $10.00 per 1M tokens
        },
        "gpt-4-turbo": {
            "input": 10.00 / 1000,  # $10.00 per 1M tokens
            "output": 30.00 / 1000,  # $30.00 per 1M tokens
        },
    },
    "anthropic": {
        "claude-3-haiku-20240307": {
            "input": 0.25 / 1000,  # $0.25 per 1M tokens
            "output": 1.25 / 1000,  # $1.25 per 1M tokens
        },
        "claude-3-5-sonnet-20241022": {
            "input": 3.00 / 1000,  # $3.00 per 1M tokens
            "output": 15.00 / 1000,  # $15.00 per 1M tokens
        },
    }
}


def estimate_tokens(text: str) -> int:
    """
    テキストのトークン数を概算（簡易版：文字数ベース）
    
    Args:
        text: テキスト
    
    Returns:
        int: 概算トークン数
    """
    # 簡易的な概算: 日本語は約2文字=1トークン、英語は約4文字=1トークン
    # 混合テキストを考慮して、平均3文字=1トークンとして概算
    if not text:
        return 0
    return len(text) // 3


def calculate_cost(
    provider: str,
    model_name: str,
    input_text: str,
    output_text: str
) -> CostInfo:
    """
    コストを計算
    
    Args:
        provider: LLMプロバイダー（"openai" or "anthropic"）
        model_name: モデル名
        input_text: 入力テキスト
        output_text: 出力テキスト
    
    Returns:
        CostInfo: コスト情報
    """
    # トークン数を概算
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)
    
    # 価格を取得（デフォルトはgpt-4o-mini相当）
    pricing = PRICING.get(provider, {}).get(model_name, PRICING["openai"]["gpt-4o-mini"])
    
    # コストを計算
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return CostInfo(
        provider=provider,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=total_cost
    )


def format_cost_info(cost_info: CostInfo) -> str:
    """
    コスト情報をフォーマット
    
    Args:
        cost_info: コスト情報
    
    Returns:
        str: フォーマットされた文字列
    """
    return (
        f"💰 コスト概算: {cost_info.estimated_cost_usd:.4f} USD "
        f"(入力: {cost_info.input_tokens} tokens, 出力: {cost_info.output_tokens} tokens, "
        f"モデル: {cost_info.model_name})"
    )


class CostTracker:
    """コスト追跡クラス"""
    
    def __init__(self):
        self.costs: list[CostInfo] = []
    
    def add_cost(self, cost_info: CostInfo):
        """コスト情報を追加"""
        self.costs.append(cost_info)
    
    def get_total_cost(self) -> float:
        """合計コストを取得"""
        return sum(cost.estimated_cost_usd for cost in self.costs)
    
    def get_summary(self) -> str:
        """コストサマリを取得"""
        if not self.costs:
            return "コスト情報なし"
        
        total = self.get_total_cost()
        count = len(self.costs)
        return f"💰 合計コスト概算: {total:.4f} USD ({count}回のLLM呼び出し)"








