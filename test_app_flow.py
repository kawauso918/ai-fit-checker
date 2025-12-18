"""
AI応募適合度チェッカー - エンドツーエンドテストスクリプト
サンプル入力で全機能の動作確認を行う
"""
import time
from datetime import datetime
from dotenv import load_dotenv

from f1_extract_requirements import extract_requirements
from f2_extract_evidence import extract_evidence
from f3_score import calculate_scores
from f4_generate_improvements import generate_improvements

# 環境変数読み込み
load_dotenv()

# サンプル求人票
sample_job_text = """
【求人票】Webエンジニア募集

■必須スキル
・Python開発経験3年以上
・Webアプリケーション開発の実務経験2年以上
・Git/GitHubを使用したチーム開発経験
・REST API設計・開発経験

■歓迎スキル
・AWSなどクラウド環境での開発・運用経験
・Docker/Kubernetesを使用したコンテナ技術の知識
・フロントエンド開発経験（React, Vue.jsなど）
・データベース設計・最適化の経験

■求める人物像
・チームでのコミュニケーションを大切にできる方
・新しい技術を積極的に学ぶ姿勢がある方
"""

# サンプル職務経歴書
sample_resume_text = """
【職務経歴書】

■職務経歴
2019年4月〜現在：株式会社ABCテクノロジー
Webエンジニア

・Pythonを使用したWebアプリケーション開発に5年間従事
・Djangoフレームワークを用いたECサイトの構築・運用
・REST APIの設計・開発・保守
・GitHubを使用したプルリクエストベースの開発フロー

■技術スタック
- 言語: Python (5年), JavaScript (3年)
- フレームワーク: Django, Flask
- データベース: PostgreSQL, MySQL
- バージョン管理: Git, GitHub
- その他: Docker基礎知識

■自己PR
チーム開発において、コードレビューを通じた品質向上に注力してきました。
また、定期的な技術勉強会を開催し、チーム全体のスキルアップに貢献しています。
"""

def main():
    print("=" * 70)
    print("AI応募適合度チェッカー - エンドツーエンドテスト")
    print("=" * 70)
    print()

    # オプション設定
    options = {
        "llm_provider": "openai",
        "model_name": None,  # デフォルトモデルを使用
        "temperature": 0.0,
        "max_must": 10,
        "max_want": 10,
        "strict_mode": False,
    }

    print(f"テスト開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"LLMプロバイダー: {options['llm_provider']}")
    print()

    start_time = time.time()

    try:
        # F1: 求人要件抽出
        print("-" * 70)
        print("F1: 求人要件抽出中...")
        print("-" * 70)
        f1_start = time.time()
        requirements = extract_requirements(sample_job_text, options)
        f1_time = time.time() - f1_start
        print(f"✅ F1完了: {len(requirements)}件の要件を抽出 ({f1_time:.2f}秒)")
        print()

        # 抽出された要件を表示
        print("【抽出された要件】")
        for req in requirements[:5]:  # 最初の5件のみ表示
            print(f"  [{req.req_id}] {req.description} ({req.category.value}, 重要度: {req.importance})")
        if len(requirements) > 5:
            print(f"  ... 他 {len(requirements) - 5}件")
        print()

        # F2: 根拠抽出
        print("-" * 70)
        print("F2: 職務経歴から根拠を抽出中...")
        print("-" * 70)
        f2_start = time.time()
        evidence_map = extract_evidence(sample_resume_text, requirements, options)
        f2_time = time.time() - f2_start
        print(f"✅ F2完了: {len(evidence_map)}件の根拠を分析 ({f2_time:.2f}秒)")
        print()

        # F3: スコア計算（強調軸なし）
        print("-" * 70)
        print("F3: スコアを計算中（強調軸なし）...")
        print("-" * 70)
        f3_start = time.time()
        score_total, score_must, score_want, matched, gaps, summary = calculate_scores(
            requirements, evidence_map
        )
        f3_time = time.time() - f3_start
        print(f"✅ F3完了: 総合スコア {score_total}点 ({f3_time:.2f}秒)")
        print()

        # スコア詳細表示
        print("【スコア詳細（強調軸なし）】")
        print(f"  総合スコア: {score_total}点")
        print(f"  Mustスコア: {score_must}点")
        print(f"  Wantスコア: {score_want}点")
        print(f"  マッチ数: {len(matched)}件")
        print(f"  ギャップ数: {len(gaps)}件")
        print()

        # F3: スコア計算（強調軸あり）
        print("-" * 70)
        print("F3: スコアを計算中（強調軸あり: 技術力, 運用）...")
        print("-" * 70)
        emphasis_axes = ["技術力", "運用"]
        score_total_with_emphasis, score_must_with_emphasis, score_want_with_emphasis, matched_with_emphasis, gaps_with_emphasis, summary_with_emphasis = calculate_scores(
            requirements, evidence_map, emphasis_axes=emphasis_axes
        )
        print(f"✅ F3完了（強調軸あり）: 総合スコア {score_total_with_emphasis}点")
        print()

        # スコア詳細表示（強調軸あり）
        print("【スコア詳細（強調軸あり）】")
        print(f"  総合スコア: {score_total_with_emphasis}点 (差分: +{score_total_with_emphasis - score_total}点)")
        print(f"  Mustスコア: {score_must_with_emphasis}点 (差分: +{score_must_with_emphasis - score_must}点)")
        print(f"  Wantスコア: {score_want_with_emphasis}点 (差分: +{score_want_with_emphasis - score_want}点)")
        print(f"  マッチ数: {len(matched_with_emphasis)}件")
        print(f"  ギャップ数: {len(gaps_with_emphasis)}件")
        print()

        # 強調軸による加点の検証
        if score_total_with_emphasis > score_total or score_must_with_emphasis > score_must or score_want_with_emphasis > score_want:
            print("✅ 強調軸による加点が正常に機能しています")
            print(f"   (Must: +{score_must_with_emphasis - score_must}点, Want: +{score_want_with_emphasis - score_want}点)")
        else:
            print("⚠️ 強調軸による加点が反映されていません")
            print("   これは、強調軸に該当する要件がマッチしていない場合に発生します")
        print()

        print(f"【総評】")
        print(f"  {summary}")
        print()

        # F4: 改善案生成
        print("-" * 70)
        print("F4: 改善案を生成中...")
        print("-" * 70)
        f4_start = time.time()
        improvements = generate_improvements(
            sample_job_text, sample_resume_text, requirements, matched, gaps, options
        )
        f4_time = time.time() - f4_start
        print(f"✅ F4完了: {len(improvements.action_items)}件の行動計画を生成 ({f4_time:.2f}秒)")
        print()

        # 改善案の概要表示
        print("【改善案概要】")
        print(f"  全体戦略: {improvements.overall_strategy[:100]}...")
        print(f"  職務経歴書編集案: {len(improvements.resume_edits)}件")
        print(f"  行動計画: {len(improvements.action_items)}件")
        if improvements.action_items:
            print("  【行動計画の一部】")
            for i, action in enumerate(improvements.action_items[:3], 1):
                print(f"    {i}. [{action.priority}] {action.action}")
        print()

        # 実行時間のサマリー
        end_time = time.time()
        total_time = end_time - start_time

        print("=" * 70)
        print("テスト完了")
        print("=" * 70)
        print(f"F1実行時間: {f1_time:.2f}秒")
        print(f"F2実行時間: {f2_time:.2f}秒")
        print(f"F3実行時間: {f3_time:.2f}秒")
        print(f"F4実行時間: {f4_time:.2f}秒")
        print(f"総実行時間: {total_time:.2f}秒")
        print()
        print("✅ 全機能が正常に動作しました！")

    except Exception as e:
        print()
        print("=" * 70)
        print("❌ エラーが発生しました")
        print("=" * 70)
        print(f"エラー内容: {e}")
        import traceback
        print()
        print("詳細なエラー情報:")
        print(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
