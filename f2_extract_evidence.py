"""
機能2: 職務経歴から根拠を抽出
"""

from models import RequirementMatch, CareerHistory


def extract_evidence(
    requirement: str,
    career_history: CareerHistory
) -> RequirementMatch:
    """
    特定の要件に対して、職務経歴から根拠となる記述を抽出
    
    Args:
        requirement: 評価対象の要件
        career_history: 職務経歴
        
    Returns:
        RequirementMatch: マッチング結果と根拠
    """
    # TODO: 実装を追加
    # AI/LLMを使用して根拠を抽出
    
    return RequirementMatch(
        requirement=requirement,
        matched=False,
        evidence="",
        confidence=0.0
    )

