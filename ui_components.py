"""
AI応募適合度チェッカー - UI表示コンポーネント
根拠表示（引用）の改善コンポーネント
"""
import streamlit as st
from typing import List
from models import (
    Requirement,
    Evidence,
    RequirementWithEvidence,
    Gap,
    RequirementType,
    MatchLevel,
    QuoteSource
)
from utils import verify_quote_in_text


def get_match_level(evidence: Evidence) -> MatchLevel:
    """
    EvidenceからMatchLevelを取得
    
    Args:
        evidence: Evidenceオブジェクト
    
    Returns:
        MatchLevel: マッチレベル
    """
    if not evidence.quotes or len(evidence.quotes) == 0:
        return MatchLevel.GAP
    
    confidence = evidence.confidence
    if confidence >= 0.7:
        return MatchLevel.MATCH
    elif confidence >= 0.4:
        return MatchLevel.PARTIAL
    else:
        return MatchLevel.GAP


def get_match_level_display(match_level: MatchLevel) -> tuple[str, str]:
    """
    MatchLevelに対応する表示ラベルと色を取得
    
    Args:
        match_level: MatchLevel
    
    Returns:
        tuple[str, str]: (ラベル, 色)
    """
    if match_level == MatchLevel.MATCH:
        return "✅ 完全一致", "green"
    elif match_level == MatchLevel.PARTIAL:
        return "⚠️ 部分一致", "orange"
    else:  # GAP
        return "❌ ギャップ", "red"


def render_requirement_with_evidence(
    requirement: Requirement,
    evidence: Evidence,
    resume_text: str,
    show_expanded: bool = False
):
    """
    要件と根拠（引用）をセットで表示
    
    Args:
        requirement: Requirementオブジェクト
        evidence: Evidenceオブジェクト
        resume_text: 職務経歴書のテキスト（引用検証用）
        show_expanded: 初期状態で展開するか
    """
    match_level = get_match_level(evidence)
    match_label, match_color = get_match_level_display(match_level)
    
    # カテゴリラベル
    category_label = "Must" if requirement.category == RequirementType.MUST else "Want"
    category_icon = "🔴" if requirement.category == RequirementType.MUST else "🟡"
    
    # Expanderのタイトル
    title = f"{category_icon} **[{requirement.req_id}]** {requirement.description}"
    
    with st.expander(title, expanded=show_expanded):
        # カテゴリと重要度
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**カテゴリ**: {category_label}**")
        with col2:
            st.markdown(f"**重要度**: {'⭐' * requirement.importance}")
        
        # マッチレベル
        st.markdown(f"**一致度**: {match_label} (信頼度: {evidence.confidence:.0%})")
        
        # 判定理由
        st.markdown("**判定理由**:")
        st.write(evidence.reason)
        
        # 引用を表示
        quotes_to_display = evidence.quotes if evidence.quotes else []
        
        if quotes_to_display:
            st.markdown("**職務経歴からの引用**:")
            for quote_obj in quotes_to_display:
                # 引用の出どころラベル
                source_label = ""
                if quote_obj.source == QuoteSource.RESUME:
                    source_label = "📄 職務経歴書"
                elif quote_obj.source == QuoteSource.RAG:
                    if quote_obj.source_id is not None and quote_obj.source_id != -1:
                        source_label = f"🔍 実績DB #{quote_obj.source_id + 1}"
                    else:
                        source_label = "🔍 実績DB"
                
                # 引用検証
                is_valid = verify_quote_in_text(quote_obj.text, resume_text)
                if is_valid:
                    st.markdown(f"> **{source_label}** {quote_obj.text}")
                else:
                    st.markdown(f"> **{source_label}** ⚠️ **引用要確認**")
                    st.markdown(f"> {quote_obj.text}")
        else:
            # 引用がない場合（GAP）
            st.markdown("**職務経歴からの引用**: なし")
            if match_level == MatchLevel.GAP:
                st.warning("⚠️ この要件に対する根拠が見つかりませんでした。職務経歴書に該当する経験を追記することを検討してください。")


def render_requirements_by_category(
    matched: List[RequirementWithEvidence],
    gaps: List[Gap],
    resume_text: str
):
    """
    要件をMust/Wantでセクション分けして表示
    
    Args:
        matched: マッチした要件と根拠のペア
        gaps: ギャップのある要件
        resume_text: 職務経歴書のテキスト（引用検証用）
    """
    # Must要件とWant要件に分類
    must_matched = [m for m in matched if m.requirement.category == RequirementType.MUST]
    want_matched = [m for m in matched if m.requirement.category == RequirementType.WANT]
    must_gaps = [g for g in gaps if g.requirement.category == RequirementType.MUST]
    want_gaps = [g for g in gaps if g.requirement.category == RequirementType.WANT]
    
    # Must要件セクション
    if must_matched or must_gaps:
        st.subheader(f"🔴 Must要件（必須）")
        
        # マッチしたMust要件
        if must_matched:
            st.markdown(f"**✅ マッチした要件（{len(must_matched)}件）**")
            for i, m in enumerate(must_matched, 1):
                render_requirement_with_evidence(
                    m.requirement,
                    m.evidence,
                    resume_text,
                    show_expanded=(i <= 3)  # 最初の3件は展開
                )
        
        # ギャップのあるMust要件
        if must_gaps:
            st.markdown(f"**❌ ギャップのある要件（{len(must_gaps)}件）**")
            for i, g in enumerate(must_gaps, 1):
                render_requirement_with_evidence(
                    g.requirement,
                    g.evidence,
                    resume_text,
                    show_expanded=(i <= 3)  # 最初の3件は展開
                )
        
        st.divider()
    
    # Want要件セクション
    if want_matched or want_gaps:
        st.subheader(f"🟡 Want要件（歓迎）")
        
        # マッチしたWant要件
        if want_matched:
            st.markdown(f"**✅ マッチした要件（{len(want_matched)}件）**")
            for i, m in enumerate(want_matched, 1):
                render_requirement_with_evidence(
                    m.requirement,
                    m.evidence,
                    resume_text,
                    show_expanded=(i <= 3)  # 最初の3件は展開
                )
        
        # ギャップのあるWant要件
        if want_gaps:
            st.markdown(f"**❌ ギャップのある要件（{len(want_gaps)}件）**")
            for i, g in enumerate(want_gaps, 1):
                render_requirement_with_evidence(
                    g.requirement,
                    g.evidence,
                    resume_text,
                    show_expanded=(i <= 3)  # 最初の3件は展開
                )
        
        st.divider()

