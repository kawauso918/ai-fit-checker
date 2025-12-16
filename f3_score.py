"""
機能3: 適合度スコアを計算
"""

from models import RequirementMatch, EvaluationResult


def calculate_score(
    must_matches: list[RequirementMatch],
    want_matches: list[RequirementMatch]
) -> EvaluationResult:
    """
    マッチング結果から適合度スコアを計算
    
    Args:
        must_matches: Must要件のマッチング結果リスト
        want_matches: Want要件のマッチング結果リスト
        
    Returns:
        EvaluationResult: 評価結果
    """
    # TODO: 実装を追加
    # スコア計算ロジック
    
    return EvaluationResult(
        must_score=0.0,
        want_score=0.0,
        total_score=0.0,
        must_matches=must_matches,
        want_matches=want_matches,
        improvements=[]
    )

