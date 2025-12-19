"""
AI応募適合度チェッカー - データモデル定義（Claude改善版）
Pydantic v2対応、LangChain PydanticOutputParser完全対応
"""
from enum import Enum
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


# ==================== Enum定義 ====================
class RequirementType(str, Enum):
    """要件の種類"""
    MUST = "Must"
    WANT = "Want"


class ConfidenceLevel(str, Enum):
    """信頼度レベル"""
    HIGH = "High"      # 0.7以上
    MEDIUM = "Medium"  # 0.4〜0.7
    LOW = "Low"        # 0.4未満
    NONE = "None"      # 0.0（マッチなし）


class QuoteSource(str, Enum):
    """引用の出どころ"""
    RESUME = "resume"  # 職務経歴書
    RAG = "rag"  # 実績メモ（RAG検索）


class Quote(BaseModel):
    """引用情報"""
    text: str = Field(..., description="引用テキスト")
    source: QuoteSource = Field(..., description="引用の出どころ")
    source_id: Optional[int] = Field(
        default=None,
        description="RAG由来の場合、実績メモの何番目のチャンク由来か（0始まり）。職務経歴書由来の場合はNone"
    )


# ==================== F1: 求人要件抽出 ====================
class Requirement(BaseModel):
    """求人票から抽出した1つの要件"""
    req_id: str = Field(..., description="要件ID（例: M1, M2, W1, W2）")
    category: RequirementType = Field(..., description="Must/Want分類")
    description: str = Field(..., description="要件の具体的な内容")
    importance: int = Field(..., ge=1, le=5, description="重要度（1=低, 5=高）")
    job_quote: str = Field(..., description="求人票からの原文引用（この要件の根拠となる部分）")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="重み（Must=1.0, Want=0.5）")

    @field_validator("req_id")
    @classmethod
    def validate_req_id(cls, v: str) -> str:
        if not v or len(v) < 2:
            raise ValueError("req_id must be at least 2 characters (e.g., M1, W1)")
        return v


class F1Output(BaseModel):
    """F1の出力形式"""
    requirements: List[Requirement] = Field(..., description="抽出した要件リスト")


# ==================== F2: 根拠引用抽出 ====================
class Evidence(BaseModel):
    """1つの要件に対する職務経歴書からの根拠"""
    req_id: str = Field(..., description="対応する要件ID")
    quotes: List[Quote] = Field(
        default_factory=list,
        description="引用リスト（職務経歴書または実績メモからの引用）。複数可。空リストはマッチなし。"
    )
    # 後方互換性のため残す（非推奨）
    resume_quotes: Optional[List[str]] = Field(
        default=None,
        description="[非推奨] 職務経歴書からの原文引用。quotesを使用してください。"
    )
    quote_sources: Optional[List[str]] = Field(
        default=None,
        description="[非推奨] 各引用の出どころ。quotesを使用してください。"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="マッチ度（0.0〜1.0）")
    confidence_level: ConfidenceLevel = Field(..., description="信頼度レベル")
    reason: str = Field(..., description="判定理由（なぜマッチ/しないか）")
    
    def __init__(self, **data):
        """後方互換性: resume_quotes/quote_sourcesからquotesを生成"""
        # 後方互換性: resume_quotes/quote_sourcesが指定されている場合、quotesに変換
        if "quotes" not in data and "resume_quotes" in data:
            resume_quotes = data.get("resume_quotes", [])
            quote_sources = data.get("quote_sources", [])
            
            if resume_quotes:
                quotes = []
                for i, quote_text in enumerate(resume_quotes):
                    source_str = quote_sources[i] if quote_sources and i < len(quote_sources) else "resume"
                    source = QuoteSource.RESUME if source_str == "resume" else QuoteSource.RAG
                    quotes.append(Quote(text=quote_text, source=source, source_id=None))
                data["quotes"] = quotes
                # 後方互換性のため、resume_quotesも残す
                if "resume_quotes" not in data:
                    data["resume_quotes"] = resume_quotes
                if "quote_sources" not in data:
                    data["quote_sources"] = quote_sources
        
        super().__init__(**data)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return v

    @field_validator("confidence_level", mode="before")
    @classmethod
    def auto_confidence_level(cls, v, info):
        """confidenceから自動的にconfidence_levelを設定"""
        if isinstance(v, str):
            return v
        confidence = info.data.get("confidence", 0.0)
        if confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif confidence > 0.0:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.NONE


class F2Output(BaseModel):
    """F2の出力形式"""
    evidence_list: List[Evidence] = Field(..., description="全要件に対する根拠リスト")


# ==================== F3: スコア計算（中間データ） ====================
class RequirementWithEvidence(BaseModel):
    """要件と根拠のペア（マッチしたもの）"""
    requirement: Requirement
    evidence: Evidence


class Gap(BaseModel):
    """ギャップのある要件"""
    requirement: Requirement
    evidence: Evidence  # confidence低い or resume_quotes空


class ScoreResult(BaseModel):
    """スコア計算結果"""
    score_total: int = Field(..., ge=0, le=100, description="総合スコア（0〜100）")
    score_must: int = Field(..., ge=0, le=100, description="Mustスコア（0〜100）")
    score_want: int = Field(..., ge=0, le=100, description="Wantスコア（0〜100）")
    matched_count: int = Field(..., ge=0, description="マッチした要件数")
    gap_count: int = Field(..., ge=0, description="ギャップのある要件数")
    summary: str = Field(..., description="スコアの総評")
    matched: List[RequirementWithEvidence] = Field(
        default_factory=list,
        description="マッチした要件と根拠のペア"
    )
    gaps: List[Gap] = Field(
        default_factory=list,
        description="ギャップのある要件"
    )


# ==================== F4: 改善案生成 ====================
class ResumeEdit(BaseModel):
    """職務経歴書への編集・追記案"""
    target_gap: str = Field(..., description="対象ギャップ（req_id または要件説明）")
    edit_type: Literal["add", "emphasize", "rewrite"] = Field(
        ...,
        description="編集タイプ（add=追記, emphasize=強調, rewrite=書き換え）"
    )
    template: str = Field(..., description="追記・編集すべき項目のテンプレート")
    example: str = Field(..., description="具体例（サンプル文章）")


class ActionItem(BaseModel):
    """行動計画の1項目"""
    priority: Literal["A", "B", "C"] = Field(
        ...,
        description="優先度（A=最優先・短期, B=中期, C=長期・余裕があれば）"
    )
    action: str = Field(..., description="具体的な行動内容")
    rationale: str = Field(..., description="なぜこの行動が有効か（根拠）")
    estimated_impact: Literal["High", "Medium", "Low"] = Field(
        ...,
        description="期待される効果（High/Medium/Low）"
    )


class Improvements(BaseModel):
    """改善案の全体"""
    resume_edits: List[ResumeEdit] = Field(
        default_factory=list,
        description="職務経歴書への編集・追記案"
    )
    action_items: List[ActionItem] = Field(
        default_factory=list,
        description="行動計画（A/B/C優先度付き）"
    )
    overall_strategy: str = Field(
        ...,
        description="全体戦略（改善の方向性を1〜2文で要約）"
    )


class F4Output(BaseModel):
    """F4の出力形式"""
    improvements: Improvements = Field(..., description="改善案")


# ==================== F5: 面接想定Q&A生成 ====================
class InterviewQA(BaseModel):
    """面接Q&Aの1項目"""
    question: str = Field(..., description="質問内容")
    answer_outline: List[str] = Field(..., description="回答の骨子（箇条書き）")


class InterviewQAs(BaseModel):
    """面接Q&Aの全体"""
    qa_list: List[InterviewQA] = Field(..., description="Q&Aリスト（10問程度）")


class F5Output(BaseModel):
    """F5の出力形式"""
    interview_qas: InterviewQAs = Field(..., description="面接Q&A")


# ==================== 実行メタ情報 ====================
class ExecutionMeta(BaseModel):
    """実行時のメタ情報"""
    model_name: str = Field(..., description="使用したLLMモデル名")
    timestamp: str = Field(..., description="実行日時（ISO8601形式）")
    f1_retry_count: int = Field(default=0, description="F1のリトライ回数")
    f2_retry_count: int = Field(default=0, description="F2のリトライ回数")
    f4_retry_count: int = Field(default=0, description="F4のリトライ回数")
    total_tokens: Optional[int] = Field(None, description="総トークン数（取得可能なら）")


# ==================== 全体の結果 ====================
class AnalysisResult(BaseModel):
    """最終的な分析結果（F1〜F4の統合 + メタ情報）"""
    requirements: List[Requirement] = Field(default_factory=list, description="F1出力")
    evidence_list: List[Evidence] = Field(default_factory=list, description="F2出力")
    score: ScoreResult = Field(..., description="F3出力")
    improvements: Improvements = Field(..., description="F4出力")
    meta: ExecutionMeta = Field(..., description="実行メタ情報")

    class Config:
        # Pydantic v2でもv1互換設定を使える
        arbitrary_types_allowed = True
        json_encoders = {
            RequirementType: lambda v: v.value,
            ConfidenceLevel: lambda v: v.value,
        }
