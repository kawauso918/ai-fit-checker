"""
F3: 適合度スコアを計算
weight加重平均 + ルールベースsummary生成 + 強調軸による加点
"""
from typing import List, Dict, Tuple, Optional
import re

from models import (
    Requirement,
    Evidence,
    RequirementWithEvidence,
    Gap,
    ScoreResult,
    RequirementType,
    ConfidenceLevel
)


# 強調軸のキーワード辞書（軸名→キーワードリスト）
EMPHASIS_KEYWORDS = {
    "技術力": ["技術", "スキル", "開発", "実装", "プログラミング", "コード", "アルゴリズム", "設計", "アーキテクチャ"],
    "セキュリティ": ["セキュリティ", "セキュア", "脆弱性", "暗号化", "認証", "認可", "セキュリティ対策", "セキュリティ監査"],
    "LLM": ["LLM", "大規模言語モデル", "GPT", "Claude", "生成AI", "AI", "機械学習", "自然言語処理", "NLP"],
    "運用": ["運用", "監視", "ログ", "デプロイ", "CI/CD", "インフラ", "サーバー", "クラウド", "AWS", "GCP", "Azure"],
    "リーダーシップ": ["リーダー", "マネジメント", "チーム", "管理", "指導", "統括", "責任者", "リード"],
    "グローバル経験": ["グローバル", "海外", "国際", "英語", "多国籍", "クロスカルチャー", "グローバルチーム"],
    "データ分析": ["データ分析", "データサイエンス", "統計", "分析", "可視化", "BI", "データウェアハウス"],
    "フロントエンド": ["フロントエンド", "UI", "UX", "React", "Vue", "Angular", "JavaScript", "TypeScript", "CSS"],
    "バックエンド": ["バックエンド", "API", "サーバー", "データベース", "マイクロサービス", "REST", "GraphQL"],
}


def calculate_scores(
    requirements: List[Requirement],
    evidence_map: Dict[str, Evidence],
    emphasis_axes: Optional[List[str]] = None
) -> Tuple[int, int, int, List[RequirementWithEvidence], List[Gap], str]:
    """
    要件と根拠からスコアを計算する（F3）

    Args:
        requirements: 要件リスト（F1の出力）
        evidence_map: req_id -> Evidence の辞書（F2の出力）
        emphasis_axes: 強調したい軸のリスト（例: ["技術力", "セキュリティ"]）

    Returns:
        tuple: (score_total, score_must, score_want, matched, gaps, summary)
            - score_total: 総合スコア（0〜100）
            - score_must: Mustスコア（0〜100）
            - score_want: Wantスコア（0〜100）
            - matched: マッチした要件と根拠のペアリスト
            - gaps: ギャップのある要件リスト
            - summary: スコアの総評（短文）
    """
    # Must/Want要件を分類
    must_requirements = [r for r in requirements if r.category == RequirementType.MUST]
    want_requirements = [r for r in requirements if r.category == RequirementType.WANT]

    # スコア計算（強調軸を渡す）
    score_must, must_matched, must_gaps = _calculate_category_score(
        must_requirements, evidence_map, emphasis_axes
    )
    score_want, want_matched, want_gaps = _calculate_category_score(
        want_requirements, evidence_map, emphasis_axes
    )

    # 総合スコア = Must*0.7 + Want*0.3（0-100にクリップ）
    score_total = int(score_must * 0.7 + score_want * 0.3)
    score_total = min(100, max(0, score_total))  # 0-100にクリップ

    # matched と gaps を統合
    matched = must_matched + want_matched
    gaps = must_gaps + want_gaps

    # サマリー生成（ルールベース）
    summary = _generate_summary(
        score_total=score_total,
        score_must=score_must,
        score_want=score_want,
        must_gap_count=len(must_gaps),
        want_gap_count=len(want_gaps),
        total_must=len(must_requirements),
        total_want=len(want_requirements)
    )

    return score_total, score_must, score_want, matched, gaps, summary


def _calculate_category_score(
    requirements: List[Requirement],
    evidence_map: Dict[str, Evidence],
    emphasis_axes: Optional[List[str]] = None
) -> Tuple[int, List[RequirementWithEvidence], List[Gap]]:
    """
    特定カテゴリ（Must or Want）のスコアを計算

    Args:
        requirements: 要件リスト
        evidence_map: req_id -> Evidence の辞書
        emphasis_axes: 強調したい軸のリスト

    Returns:
        tuple: (score, matched, gaps)
            - score: カテゴリスコア（0〜100）
            - matched: マッチした要件と根拠のペア
            - gaps: ギャップのある要件
    """
    if not requirements:
        return 100, [], []  # 要件がない場合は満点

    matched = []
    gaps = []
    total_weighted_score = 0.0
    total_weight = 0.0

    for req in requirements:
        # 対応するEvidenceを取得
        evidence = evidence_map.get(req.req_id)

        if not evidence:
            # Evidenceが存在しない場合（本来はF2で補完されているはず）
            evidence = Evidence(
                req_id=req.req_id,
                resume_quotes=[],
                confidence=0.0,
                confidence_level=ConfidenceLevel.NONE,
                reason="根拠が見つかりませんでした"
            )

        # Confidence点数化
        # strong (≥0.7) = 1.0, partial (0.4-0.7) = 0.5, none (<0.4) = 0.0
        if evidence.confidence >= 0.7:
            points = 1.0
        elif evidence.confidence >= 0.4:
            points = 0.5
        else:
            points = 0.0

        # 強調軸による加点（該当要件のみ）
        if emphasis_axes and points > 0.0:  # マッチしている要件のみ加点
            bonus = _calculate_emphasis_bonus(req, emphasis_axes)
            points = min(1.0, points + bonus)  # 最大1.0にクリップ

        # Weight加重平均
        weight = req.weight
        total_weighted_score += points * weight
        total_weight += weight

        # Matched / Gap 判定
        if points > 0.0:  # strong または partial
            matched.append(RequirementWithEvidence(
                requirement=req,
                evidence=evidence
            ))
        else:  # none
            gaps.append(Gap(
                requirement=req,
                evidence=evidence
            ))

    # カテゴリスコア計算（加重平均 * 100）
    if total_weight > 0:
        score = int((total_weighted_score / total_weight) * 100)
        score = min(100, score)  # 0-100にクリップ
    else:
        score = 0

    return score, matched, gaps


def _calculate_emphasis_bonus(requirement: Requirement, emphasis_axes: List[str]) -> float:
    """
    強調軸に基づいて加点を計算

    Args:
        requirement: 要件
        emphasis_axes: 強調したい軸のリスト

    Returns:
        float: 加点値（最大0.1程度）
    """
    if not emphasis_axes:
        return 0.0

    # 要件の説明と引用を結合して検索対象にする
    search_text = (requirement.description + " " + requirement.job_quote).lower()

    # 各強調軸についてキーワードマッチを確認
    matched_axes = []
    for axis in emphasis_axes:
        axis = axis.strip()
        if not axis:
            continue
        
        # キーワード辞書から取得、または軸名そのものをキーワードとして使用
        keywords = EMPHASIS_KEYWORDS.get(axis, [axis])
        
        # いずれかのキーワードが含まれているか確認
        for keyword in keywords:
            if keyword.lower() in search_text:
                matched_axes.append(axis)
                break

    # マッチした軸数に応じて加点（最大0.1）
    if matched_axes:
        # 1軸マッチで0.05、2軸以上で0.1
        bonus = min(0.1, 0.05 * len(matched_axes))
        return bonus

    return 0.0


def _generate_summary(
    score_total: int,
    score_must: int,
    score_want: int,
    must_gap_count: int,
    want_gap_count: int,
    total_must: int,
    total_want: int
) -> str:
    """
    スコアからサマリーを生成（ルールベース）

    Args:
        score_total: 総合スコア
        score_must: Mustスコア
        score_want: Wantスコア
        must_gap_count: Must要件のギャップ数
        want_gap_count: Want要件のギャップ数
        total_must: Must要件総数
        total_want: Want要件総数

    Returns:
        str: サマリー文
    """
    # スコアレベル判定
    if score_total >= 80:
        level = "非常に高い"
    elif score_total >= 60:
        level = "高い"
    elif score_total >= 40:
        level = "中程度"
    else:
        level = "低い"

    # Must要件の充足状況
    if must_gap_count == 0:
        must_status = "全てのMust要件を満たしています"
    elif must_gap_count == 1:
        must_status = f"Must要件のうち1件が不足しています"
    else:
        must_status = f"Must要件のうち{must_gap_count}件が不足しています"

    # Want要件の充足状況
    if total_want == 0:
        want_status = ""
    elif want_gap_count == 0:
        want_status = "。Want要件も全て満たしています"
    else:
        want_matched = total_want - want_gap_count
        want_status = f"。Want要件は{total_want}件中{want_matched}件を満たしています"

    # サマリー組み立て
    summary = f"総合適合度は{level}です（{score_total}点）。{must_status}{want_status}。"

    # 改善提案の追加
    if must_gap_count > 0:
        summary += f" Must要件の不足を埋めることを最優先に検討してください。"
    elif score_total < 80:
        summary += f" Want要件を強化することで、さらに適合度を高められます。"

    return summary


def get_score_result(
    requirements: List[Requirement],
    evidence_map: Dict[str, Evidence],
    emphasis_axes: Optional[List[str]] = None
) -> ScoreResult:
    """
    ScoreResult形式でスコアを返す（便利関数）

    Args:
        requirements: 要件リスト
        evidence_map: req_id -> Evidence の辞書
        emphasis_axes: 強調したい軸のリスト

    Returns:
        ScoreResult: スコア計算結果
    """
    score_total, score_must, score_want, matched, gaps, summary = calculate_scores(
        requirements, evidence_map, emphasis_axes
    )

    return ScoreResult(
        score_total=score_total,
        score_must=score_must,
        score_want=score_want,
        matched_count=len(matched),
        gap_count=len(gaps),
        summary=summary,
        matched=matched,
        gaps=gaps
    )


# ==================== テスト用コード ====================
if __name__ == "__main__":
    from f1_extract_requirements import extract_requirements
    from f2_extract_evidence import extract_evidence

    # サンプル求人票
    sample_job_text = """
【求人票】Webエンジニア募集

■必須スキル
・Python開発経験3年以上
・Webアプリケーション開発の実務経験

■歓迎スキル
・AWSなどクラウド環境での開発経験
・機械学習・データ分析の知識
    """

    # サンプル職務経歴書
    sample_resume_text = """
【職務経歴書】

■職務経歴
2019年〜現在：株式会社ABC
・Pythonを使用したWebアプリケーション開発に5年間従事
・Djangoフレームワークを用いたECサイトの構築
・AWS (EC2, S3, RDS) を活用したインフラ構築

■スキル
・Python（5年）、JavaScript（3年）
・Django, Flask, FastAPI
・AWS, Docker, Git
    """

    print("=" * 60)
    print("F1→F2→F3 統合テスト")
    print("=" * 60)

    try:
        # F1: 要件抽出
        print("\n[Step 1] F1: 求人要件抽出")
        requirements = extract_requirements(
            job_text=sample_job_text,
            options={"max_must": 3, "max_want": 3}
        )
        print(f"✅ {len(requirements)}件抽出")

        # F2: 根拠抽出
        print("\n[Step 2] F2: 根拠抽出")
        evidence_map = extract_evidence(
            resume_text=sample_resume_text,
            requirements=requirements
        )
        print(f"✅ {len(evidence_map)}件分析")

        # F3: スコア計算
        print("\n[Step 3] F3: スコア計算")
        score_total, score_must, score_want, matched, gaps, summary = calculate_scores(
            requirements=requirements,
            evidence_map=evidence_map
        )

        print(f"\n{'='*60}")
        print("📊 スコア結果")
        print(f"{'='*60}")
        print(f"総合スコア: {score_total}点")
        print(f"  ├─ Mustスコア: {score_must}点")
        print(f"  └─ Wantスコア: {score_want}点")
        print()
        print(f"マッチ数: {len(matched)}件")
        print(f"ギャップ数: {len(gaps)}件")
        print()
        print(f"【総評】\n{summary}")
        print()

        # マッチ詳細
        if matched:
            print(f"\n✅ マッチした要件（{len(matched)}件）")
            for m in matched:
                print(f"  [{m.requirement.req_id}] {m.requirement.description}")
                print(f"    → Confidence: {m.evidence.confidence:.2f} ({m.evidence.confidence_level.value})")

        # ギャップ詳細
        if gaps:
            print(f"\n⚠️  ギャップのある要件（{len(gaps)}件）")
            for g in gaps:
                print(f"  [{g.requirement.req_id}] {g.requirement.description}")
                print(f"    → {g.evidence.reason}")

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
