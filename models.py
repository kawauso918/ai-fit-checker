"""
データモデル定義
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class JobRequirement:
    """求人要件"""
    must_requirements: List[str]  # 必須要件
    want_requirements: List[str]  # 歓迎要件
    job_description: str  # 職務内容
    company_info: Optional[str] = None  # 会社情報


@dataclass
class CareerHistory:
    """職務経歴"""
    experiences: List[str]  # 職務経験
    skills: List[str]  # スキル
    achievements: Optional[List[str]] = None  # 実績


@dataclass
class RequirementMatch:
    """要件マッチング結果"""
    requirement: str  # 要件内容
    matched: bool  # マッチしたかどうか
    evidence: str  # 根拠となる記述
    confidence: float  # 信頼度 (0.0-1.0)


@dataclass
class EvaluationResult:
    """評価結果"""
    must_score: float  # Must要件スコア (0.0-1.0)
    want_score: float  # Want要件スコア (0.0-1.0)
    total_score: float  # 総合スコア (0.0-1.0)
    must_matches: List[RequirementMatch]  # Must要件マッチング結果
    want_matches: List[RequirementMatch]  # Want要件マッチング結果
    improvements: List[str]  # 改善案

