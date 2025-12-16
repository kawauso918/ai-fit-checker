"""
機能1: 求人票から要件を抽出
"""

from models import JobRequirement


def extract_requirements(job_post_text: str) -> JobRequirement:
    """
    求人票のテキストから必須要件と歓迎要件を抽出
    
    Args:
        job_post_text: 求人票のテキスト
        
    Returns:
        JobRequirement: 抽出された要件
    """
    # TODO: 実装を追加
    # AI/LLMを使用して要件を抽出
    
    return JobRequirement(
        must_requirements=[],
        want_requirements=[],
        job_description=""
    )

