"""
AI応募適合度チェッカー - 入力検証
求人票/職務経歴の入力検証とエラーメッセージ生成
"""
from typing import Tuple, Optional, List
from models import Requirement


def validate_job_text(job_text: str) -> Tuple[bool, Optional[str]]:
    """
    求人票のテキストを検証
    
    Args:
        job_text: 求人票のテキスト
    
    Returns:
        Tuple[bool, Optional[str]]: (検証成功, エラーメッセージ)
            - 検証成功: (True, None)
            - 検証失敗: (False, エラーメッセージ)
    """
    if not job_text or not job_text.strip():
        return False, "求人票が入力されていません。求人票のテキストを貼り付けてください。"
    
    if len(job_text.strip()) < 300:
        return False, (
            "求人票が短すぎます（300文字未満）。\n\n"
            "以下の情報を含む箇所を追加で貼り付けてください：\n"
            "- 募集要件・必須条件\n"
            "- 歓迎要件・希望スキル\n"
            "- 仕事内容・業務内容\n"
            "- 必要な経験・スキル\n\n"
            "これらの情報がある部分を追加すると、より正確な分析が可能です。"
        )
    
    return True, None


def validate_resume_text(resume_text: str) -> Tuple[bool, Optional[str]]:
    """
    職務経歴書のテキストを検証
    
    Args:
        resume_text: 職務経歴書のテキスト
    
    Returns:
        Tuple[bool, Optional[str]]: (検証成功, エラーメッセージ)
            - 検証成功: (True, None)
            - 検証失敗: (False, エラーメッセージ)
    """
    if not resume_text or not resume_text.strip():
        return False, "職務経歴書が入力されていません。職務経歴書のテキストを貼り付けてください。"
    
    if len(resume_text.strip()) < 200:
        return False, (
            "職務経歴書が短すぎます（200文字未満）。\n\n"
            "以下の情報を追記してください：\n"
            "- 担当業務・職務内容\n"
            "- 使用技術・スキル\n"
            "- 成果・実績\n"
            "- 期間・在籍期間\n\n"
            "これらの情報を追加すると、より正確な分析が可能です。"
        )
    
    return True, None


def validate_requirements_extracted(requirements: List[Requirement]) -> Tuple[bool, Optional[str]]:
    """
    要件抽出結果を検証（求人が曖昧な場合）
    
    Args:
        requirements: 抽出された要件リスト
    
    Returns:
        Tuple[bool, Optional[str]]: (検証成功, エラーメッセージ)
            - 検証成功: (True, None)
            - 検証失敗: (False, エラーメッセージ)
    """
    if not requirements or len(requirements) == 0:
        return False, (
            "求人票から要件を抽出できませんでした。\n\n"
            "以下の情報を含む箇所を追加で貼り付けてください：\n"
            "- 必須スキル・必須条件\n"
            "- 歓迎スキル・希望条件\n"
            "- 仕事内容・業務内容\n"
            "- 必要な経験・スキル\n\n"
            "これらの情報がある部分を追加すると、要件抽出が可能になります。"
        )
    
    # Must要件が0件の場合も警告
    must_count = sum(1 for r in requirements if r.category.value == "Must")
    if must_count == 0:
        return False, (
            "必須要件（Must）が抽出できませんでした。\n\n"
            "求人票に「必須スキル」「必須条件」「必須要件」などの記載がある部分を"
            "追加で貼り付けてください。"
        )
    
    return True, None


def validate_inputs(job_text: str, resume_text: str) -> Tuple[bool, Optional[str]]:
    """
    入力全体を検証（求人票 + 職務経歴書）
    
    Args:
        job_text: 求人票のテキスト
        resume_text: 職務経歴書のテキスト
    
    Returns:
        Tuple[bool, Optional[str]]: (検証成功, エラーメッセージ)
            - 検証成功: (True, None)
            - 検証失敗: (False, エラーメッセージ)
    """
    # 求人票の検証
    job_valid, job_error = validate_job_text(job_text)
    if not job_valid:
        return False, job_error
    
    # 職務経歴書の検証
    resume_valid, resume_error = validate_resume_text(resume_text)
    if not resume_valid:
        return False, resume_error
    
    return True, None

