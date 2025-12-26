"""
分析結果・応募メール下書き・チャット履歴のエクスポート機能
Markdown / テキスト形式でダウンロード可能にする
"""
from typing import List, Dict, Any, Optional
from datetime import datetime

from models import (
    Requirement,
    RequirementWithEvidence,
    Gap,
    Improvements,
    EmailDraft,
    RequirementType
)


# 個人情報に関する注意書き
PERSONAL_INFO_WARNING = """⚠️ **個人情報に関する注意**
本ファイルには職務経歴書や求人票の情報が含まれている可能性があります。
個人情報や機密情報が含まれていないか確認し、適切に管理してください。
"""


def export_analysis_to_md(result_dict: Dict[str, Any]) -> str:
    """
    分析結果をMarkdown形式でエクスポート
    
    Args:
        result_dict: 分析結果の辞書
            - score_total, score_must, score_want
            - matched (List[RequirementWithEvidence])
            - gaps (List[Gap])
            - summary (str)
            - improvements (Improvements)
            - requirements (List[Requirement])
            - timestamp (str)
            - execution_time (float)
    
    Returns:
        str: Markdown形式の文字列
    """
    lines = []
    
    # ヘッダー
    lines.append("# AI応募適合度チェッカー - 分析結果")
    lines.append("")
    lines.append(f"**生成日時**: {result_dict.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")
    if result_dict.get('execution_time'):
        lines.append(f"**実行時間**: {result_dict.get('execution_time', 0):.2f}秒")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 個人情報の注意書き
    lines.append(PERSONAL_INFO_WARNING)
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 総合スコア
    lines.append("# 総合スコア")
    lines.append("")
    score_total = result_dict.get('score_total', 0)
    score_must = result_dict.get('score_must', 0)
    score_want = result_dict.get('score_want', 0)
    
    lines.append(f"- **総合スコア**: {score_total}点")
    lines.append(f"- **Mustスコア**: {score_must}点")
    lines.append(f"- **Wantスコア**: {score_want}点")
    lines.append("")
    
    # サマリ
    summary = result_dict.get('summary', '')
    if summary:
        lines.append("## サマリ")
        lines.append("")
        lines.append(summary)
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # 要件一覧（Must/Want）
    requirements = result_dict.get('requirements', [])
    if requirements:
        lines.append("# 抽出された要件")
        lines.append("")
        
        # Must要件
        must_requirements = [r for r in requirements if r.category == RequirementType.MUST]
        if must_requirements:
            lines.append("## Must要件（必須）")
            lines.append("")
            for i, req in enumerate(must_requirements, 1):
                lines.append(f"### {i}. [{req.req_id}] {req.description}")
                if req.category:
                    lines.append(f"- **カテゴリ**: {req.category.value}")
                lines.append("")
        
        # Want要件
        want_requirements = [r for r in requirements if r.category == RequirementType.WANT]
        if want_requirements:
            lines.append("## Want要件（歓迎）")
            lines.append("")
            for i, req in enumerate(want_requirements, 1):
                lines.append(f"### {i}. [{req.req_id}] {req.description}")
                if req.category:
                    lines.append(f"- **カテゴリ**: {req.category.value}")
                lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # 一致した要件（強み）
    matched = result_dict.get('matched', [])
    if matched:
        lines.append("# 一致した要件（強み）")
        lines.append("")
        
        for i, m in enumerate(matched, 1):
            req = m.requirement
            evidence = m.evidence
            
            lines.append(f"## {i}. [{req.req_id}] {req.description}")
            lines.append("")
            lines.append(f"- **一致度**: {evidence.confidence:.0%}")
            lines.append(f"- **要件タイプ**: {req.category.value.upper()}")
            if req.category:
                lines.append(f"- **カテゴリ**: {req.category.value}")
            lines.append("")
            
            # 引用
            if evidence.quotes:
                lines.append("### 根拠（引用）")
                lines.append("")
                for j, quote in enumerate(evidence.quotes, 1):
                    # quote.sourceはQuoteSource Enum
                    quote_source = quote.source.value if hasattr(quote.source, 'value') else str(quote.source)
                    if quote_source == "resume":
                        quote_source_label = "職務経歴書"
                    elif quote_source == "rag":
                        if hasattr(quote, 'source_id') and quote.source_id is not None:
                            quote_source_label = f"実績DB #{quote.source_id}"
                        else:
                            quote_source_label = "実績DB"
                    else:
                        quote_source_label = quote_source
                    lines.append(f"**引用{j}** ({quote_source_label}):")
                    lines.append("")
                    lines.append(f"> {quote.text}")
                    lines.append("")
            
            lines.append("---")
            lines.append("")
    
    # 不足している要件（ギャップ）
    gaps = result_dict.get('gaps', [])
    if gaps:
        lines.append("# 不足している要件（ギャップ）")
        lines.append("")
        
        for i, gap in enumerate(gaps, 1):
            req = gap.requirement
            
            lines.append(f"## {i}. [{req.req_id}] {req.description}")
            lines.append("")
            lines.append(f"- **要件タイプ**: {req.category.value.upper()}")
            if req.category:
                lines.append(f"- **カテゴリ**: {req.category.value}")
            if gap.reason:
                lines.append(f"- **不足理由**: {gap.reason}")
            if gap.improvement_direction:
                lines.append(f"- **改善の方向性**: {gap.improvement_direction}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # 改善案
    improvements = result_dict.get('improvements')
    if improvements:
        lines.append("# 改善案")
        lines.append("")
        
        if improvements.overall_strategy:
            lines.append("## 全体戦略")
            lines.append("")
            lines.append(improvements.overall_strategy)
            lines.append("")
        
        if improvements.resume_edits:
            lines.append("## 職務経歴書の編集・追記案")
            lines.append("")
            for i, edit in enumerate(improvements.resume_edits, 1):
                lines.append(f"### {i}. {edit.title}")
                if edit.template:
                    lines.append("**テンプレート**:")
                    lines.append("")
                    lines.append("```")
                    lines.append(edit.template)
                    lines.append("```")
                    lines.append("")
                if edit.example:
                    lines.append("**具体例**:")
                    lines.append("")
                    lines.append("```")
                    lines.append(edit.example)
                    lines.append("```")
                    lines.append("")
        
        if improvements.action_plans:
            lines.append("## 行動計画")
            lines.append("")
            for plan in improvements.action_plans:
                priority = plan.priority.value if hasattr(plan.priority, 'value') else plan.priority
                lines.append(f"### [{priority}] {plan.title}")
                lines.append("")
                if plan.description:
                    lines.append(plan.description)
                    lines.append("")
                if plan.steps:
                    for step in plan.steps:
                        lines.append(f"- {step}")
                    lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # 次アクション
    lines.append("# 次アクション")
    lines.append("")
    lines.append("1. 改善案を参考に職務経歴書を更新")
    lines.append("2. 応募メール下書きを作成")
    lines.append("3. 面接想定Q&Aを準備")
    lines.append("4. 最終確認（誤字脱字、個人情報のマスク）")
    lines.append("")
    
    return "\n".join(lines)


def export_email_to_txt(email_draft: EmailDraft) -> str:
    """
    応募メール下書きをテキスト形式でエクスポート
    
    Args:
        email_draft: EmailDraftオブジェクト
    
    Returns:
        str: テキスト形式の文字列
    """
    lines = []
    
    # ヘッダー
    lines.append("=" * 60)
    lines.append("応募メール下書き")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("")
    
    # 個人情報の注意書き
    lines.append(PERSONAL_INFO_WARNING.replace("**", "").replace("⚠️", "⚠"))
    lines.append("")
    lines.append("-" * 60)
    lines.append("")
    
    # 件名案
    lines.append("【件名案】")
    lines.append("")
    for i, subject in enumerate(email_draft.subject_options, 1):
        lines.append(f"{i}. {subject}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("")
    
    # 本文
    lines.append("【本文】")
    lines.append("")
    lines.append(email_draft.body)
    lines.append("")
    lines.append("-" * 60)
    lines.append("")
    
    # 根拠リスト
    if email_draft.evidence_list:
        lines.append("【根拠リスト】")
        lines.append("")
        for i, evidence in enumerate(email_draft.evidence_list, 1):
            lines.append(f"{i}. 主張: {evidence.claim}")
            lines.append(f"   根拠タイプ: {evidence.evidence_type}")
            if evidence.requirement_id:
                lines.append(f"   対応要件ID: {evidence.requirement_id}")
            lines.append(f"   根拠テキスト: {evidence.evidence_text}")
            lines.append("")
        lines.append("-" * 60)
        lines.append("")
    
    # 注意事項
    if email_draft.notes:
        lines.append("【注意事項】")
        lines.append("")
        for i, note in enumerate(email_draft.notes, 1):
            lines.append(f"{i}. {note}")
        lines.append("")
    
    # 最終確認
    lines.append("-" * 60)
    lines.append("")
    lines.append("【送信前の確認事項】")
    lines.append("")
    lines.append("□ 誤字脱字がないか確認")
    lines.append("□ 企業名・役職名が正しいか確認")
    lines.append("□ 職務経歴にない経験を断定していないか確認")
    lines.append("□ 個人情報が含まれていないか確認")
    lines.append("")
    
    return "\n".join(lines)


def export_chat_to_md(chat_history: List[tuple], mode: str = "default") -> str:
    """
    求人深掘りチャット履歴をMarkdown形式でエクスポート
    
    Args:
        chat_history: チャット履歴のリスト [(user_message, assistant_response), ...]
        mode: モード名（ファイル名に使用）
            - "job_understanding": 求人理解
            - "email_improvement": 応募メール改善
            - "interview_questions": 面接質問作成
    
    Returns:
        str: Markdown形式の文字列
    """
    lines = []
    
    # モード名の表示名
    mode_display_names = {
        "job_understanding": "📖 求人理解",
        "email_improvement": "📧 応募メール改善",
        "interview_questions": "❓ 面接質問作成",
        "default": "チャット"
    }
    mode_display = mode_display_names.get(mode, mode)
    
    # ヘッダー
    lines.append("# 求人深掘りチャット履歴")
    lines.append("")
    lines.append(f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**モード**: {mode_display}")
    lines.append(f"**会話数**: {len(chat_history)}件")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 個人情報の注意書き
    lines.append(PERSONAL_INFO_WARNING)
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # チャット履歴
    if not chat_history:
        lines.append("チャット履歴がありません。")
        lines.append("")
    else:
        for i, (user_msg, assistant_msg) in enumerate(chat_history, 1):
            lines.append(f"## 会話 {i}")
            lines.append("")
            lines.append("### あなた")
            lines.append("")
            lines.append(user_msg)
            lines.append("")
            lines.append("### アシスタント")
            lines.append("")
            lines.append(assistant_msg)
            lines.append("")
            lines.append("---")
            lines.append("")
    
    return "\n".join(lines)

