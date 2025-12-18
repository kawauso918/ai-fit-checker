"""
AI応募適合度チェッカー - PDFエクスポート機能
分析結果をPDF形式でダウンロード可能にする
"""
from io import BytesIO
from typing import Dict, List
from fpdf import FPDF


def _safe_encode(text: str) -> str:
    """
    テキストを安全にエンコード（日本語文字をASCII互換に変換）
    
    Args:
        text: エンコード対象のテキスト
    
    Returns:
        str: エンコード後のテキスト
    """
    if not text:
        return ""
    
    # 日本語文字をASCII互換の表現に変換（簡易版）
    # 実際の実装では、より高度な変換が必要な場合がある
    try:
        # UTF-8でエンコードしてからデコード（エラー処理）
        return text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    except Exception:
        # エラー時は元のテキストを返す
        return str(text)


class PDFReport(FPDF):
    """PDFレポート生成クラス"""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
        self.add_page()
    
    def header(self):
        """ヘッダー"""
        self.set_font("Arial", "B", 16)
        title = _safe_encode("AI Application Fit Checker - Analysis Result")
        self.cell(0, 10, title, 0, 1, "C")
        self.ln(5)
    
    def footer(self):
        """フッター"""
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")
    
    def add_section_title(self, title: str):
        """セクションタイトルを追加"""
        self.ln(5)
        self.set_font("Arial", "B", 12)
        safe_title = _safe_encode(title)
        self.cell(0, 10, safe_title, 0, 1)
        self.ln(2)
    
    def add_text(self, text: str, font_size=10, style=""):
        """テキストを追加（改行対応）"""
        self.set_font("Arial", style, font_size)
        safe_text = _safe_encode(text)
        # テキストを適切な幅で折り返し
        lines = self._wrap_text(safe_text, 180)  # 幅180mmで折り返し
        for line in lines:
            try:
                self.cell(0, 6, line, 0, 1)
            except Exception:
                # エラー時は空行を追加
                self.ln(6)
    
    def add_multicell(self, text: str, font_size=10, style=""):
        """複数行テキストを追加"""
        self.set_font("Arial", style, font_size)
        safe_text = _safe_encode(text)
        try:
            self.multi_cell(0, 6, safe_text)
        except Exception:
            # エラー時は空行を追加
            self.ln(6)
        self.ln(2)
    
    def _wrap_text(self, text: str, width_mm: float) -> List[str]:
        """テキストを指定幅で折り返し"""
        if not text:
            return [""]
        
        # 簡易的な折り返し処理（文字数ベース）
        # 1mm ≈ 0.4文字（Arial 10ptの場合）
        max_chars = int(width_mm * 0.4)
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines if lines else [text]


def generate_pdf(result: Dict) -> BytesIO:
    """
    分析結果をPDF形式で生成
    
    Args:
        result: 分析結果の辞書
    
    Returns:
        BytesIO: PDFデータ
    """
    pdf = PDFReport()
    
    # 個人情報注意書き
    pdf.add_section_title("Important: Personal Information Handling")
    pdf.add_multicell(
        "This report was generated using LLM (Large Language Model). "
        "It may contain personal information (name, address, phone number, etc.). "
        "Please handle with care.",
        font_size=9,
        style="I"
    )
    pdf.ln(5)
    
    # 実行情報
    pdf.add_section_title("Execution Information")
    pdf.add_text(f"Timestamp: {result.get('timestamp', 'N/A')}", font_size=9)
    pdf.add_text(f"Execution Time: {result.get('execution_time', 0):.2f} seconds", font_size=9)
    pdf.ln(3)
    
    # スコア
    pdf.add_section_title("Scores")
    pdf.add_text(f"Total Score: {result.get('score_total', 0)} points", font_size=11, style="B")
    pdf.add_text(f"Must Score: {result.get('score_must', 0)} points", font_size=10)
    pdf.add_text(f"Want Score: {result.get('score_want', 0)} points", font_size=10)
    pdf.ln(3)
    
    # 総評
    pdf.add_section_title("Summary")
    pdf.add_multicell(result.get('summary', ''), font_size=10)
    pdf.ln(3)
    
    # 差分サマリ（強みTop3 / ギャップTop3）
    matched = result.get('matched', [])
    gaps = result.get('gaps', [])
    
    # 強みTop3を抽出（簡易版：confidence順）
    top_strengths = sorted(matched, key=lambda m: -m.evidence.confidence)[:3]
    if top_strengths:
        pdf.add_section_title("Top 3 Strengths")
        for i, m in enumerate(top_strengths, 1):
            category_label = "Must" if m.requirement.category.value == "Must" else "Want"
            pdf.add_text(
                f"{i}. {m.requirement.description} ({category_label}, Match: {m.evidence.confidence:.0%})",
                font_size=9
            )
        pdf.ln(3)
    
    # 致命的ギャップTop3を抽出（Must優先）
    must_gaps = [g for g in gaps if g.requirement.category.value == "Must"]
    want_gaps = [g for g in gaps if g.requirement.category.value == "Want"]
    top_gaps = (must_gaps + want_gaps)[:3]
    
    if top_gaps:
        pdf.add_section_title("Top 3 Critical Gaps")
        for i, g in enumerate(top_gaps, 1):
            category_label = "Must" if g.requirement.category.value == "Must" else "Want"
            pdf.add_text(
                f"{i}. {g.requirement.description} ({category_label})",
                font_size=9
            )
        pdf.ln(3)
    
    # マッチした要件
    if matched:
        pdf.add_section_title(f"Matched Requirements ({len(matched)} items)")
        for i, m in enumerate(matched, 1):
            pdf.add_text(
                f"[{m.requirement.req_id}] {m.requirement.description}",
                font_size=10,
                style="B"
            )
            pdf.add_text(
                f"Match: {m.evidence.confidence:.2f} ({m.evidence.confidence_level.value})",
                font_size=9
            )
            
            if m.evidence.resume_quotes:
                pdf.add_text("Quotes from Resume:", font_size=9, style="I")
                for quote in m.evidence.resume_quotes:
                    pdf.add_multicell(f"  > {quote}", font_size=8)
            
            pdf.ln(2)
        pdf.ln(3)
    
    # ギャップのある要件
    if gaps:
        pdf.add_section_title(f"Gap Requirements ({len(gaps)} items)")
        for i, g in enumerate(gaps, 1):
            pdf.add_text(
                f"[{g.requirement.req_id}] {g.requirement.description}",
                font_size=10,
                style="B"
            )
            pdf.add_text(
                f"Category: {g.requirement.category.value}",
                font_size=9
            )
            pdf.add_multicell(f"Gap Reason: {g.evidence.reason}", font_size=9)
            pdf.ln(2)
        pdf.ln(3)
    
    # 改善案
    improvements = result.get('improvements')
    if improvements:
        pdf.add_section_title("Improvement Suggestions")
        
        # 全体戦略
        pdf.add_text("[Overall Strategy]", font_size=10, style="B")
        pdf.add_multicell(improvements.overall_strategy, font_size=9)
        pdf.ln(3)
        
        # 職務経歴書の編集・追記案
        if improvements.resume_edits:
            pdf.add_text(
                f"Resume Edit Suggestions ({len(improvements.resume_edits)} items)",
                font_size=10,
                style="B"
            )
            pdf.ln(2)
            
            for i, edit in enumerate(improvements.resume_edits, 1):
                pdf.add_text(
                    f"{i}. Target: {edit.target_gap} ({edit.edit_type})",
                    font_size=9,
                    style="B"
                )
                pdf.add_text("Template:", font_size=9, style="I")
                pdf.add_multicell(edit.template, font_size=8)
                pdf.add_text("Example:", font_size=9, style="I")
                pdf.add_multicell(edit.example, font_size=8)
                pdf.ln(2)
            pdf.ln(3)
        
        # 行動計画
        if improvements.action_items:
            pdf.add_text(
                f"Action Plan ({len(improvements.action_items)} items)",
                font_size=10,
                style="B"
            )
            pdf.ln(2)
            
            # 優先度別にグループ化
            priority_a = [a for a in improvements.action_items if a.priority == "A"]
            priority_b = [a for a in improvements.action_items if a.priority == "B"]
            priority_c = [a for a in improvements.action_items if a.priority == "C"]
            
            if priority_a:
                pdf.add_text("Priority A (High Priority, Short-term)", font_size=9, style="B")
                for a in priority_a:
                    pdf.add_text(f"- {a.action}", font_size=9)
                    pdf.add_multicell(f"  Rationale: {a.rationale}", font_size=8)
                    pdf.add_multicell(f"  Expected Impact: {a.estimated_impact}", font_size=8)
                pdf.ln(2)
            
            if priority_b:
                pdf.add_text("Priority B (Medium-term)", font_size=9, style="B")
                for a in priority_b:
                    pdf.add_text(f"- {a.action}", font_size=9)
                    pdf.add_multicell(f"  Rationale: {a.rationale}", font_size=8)
                    pdf.add_multicell(f"  Expected Impact: {a.estimated_impact}", font_size=8)
                pdf.ln(2)
            
            if priority_c:
                pdf.add_text("Priority C (Long-term, Optional)", font_size=9, style="B")
                for a in priority_c:
                    pdf.add_text(f"- {a.action}", font_size=9)
                    pdf.add_multicell(f"  Rationale: {a.rationale}", font_size=8)
                    pdf.add_multicell(f"  Expected Impact: {a.estimated_impact}", font_size=8)
                pdf.ln(2)
    
    # PDFをBytesIOに出力
    pdf_bytes = BytesIO()
    pdf.output(pdf_bytes)
    pdf_bytes.seek(0)
    
    return pdf_bytes

