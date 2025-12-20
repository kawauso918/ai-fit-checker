"""
スモークテスト: F1〜F6の基本動作確認
RAGなし/あり両方でテスト
"""
import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from f1_extract_requirements import extract_requirements
from f2_extract_evidence import extract_evidence
from f3_score import calculate_scores
from f4_generate_improvements import generate_improvements
from f5_generate_interview_qa import generate_interview_qa
from f6_quality_evaluation import evaluate_quality


# サンプルデータ
SAMPLE_JOB_TEXT = """
【求人票】Webエンジニア募集

【職務内容】
- Webアプリケーションの開発・保守
- API設計・実装
- データベース設計・最適化

【必須要件】
- Python 3年以上の実務経験
- DjangoまたはFlaskの開発経験
- PostgreSQLまたはMySQLの使用経験
- Git/GitHubの使用経験

【歓迎要件】
- AWS等のクラウドサービスの使用経験
- Docker/Kubernetesの使用経験
- CI/CDパイプラインの構築経験
"""

SAMPLE_RESUME_TEXT = """
【職務経歴】

■ 株式会社ABC（2020年4月 - 現在）
職務：バックエンドエンジニア

主な業務内容：
- Pythonを使用したWebアプリケーション開発（Django）
- RESTful APIの設計・実装
- PostgreSQLデータベースの設計・運用
- コードレビューと技術的負債の改善

使用技術：
- Python, Django, PostgreSQL, Git, GitHub
- AWS (EC2, S3, RDS)
- Docker

実績：
- レスポンスタイムを50%改善
- テストカバレッジを80%に向上
"""

SAMPLE_ACHIEVEMENT_NOTES = """
【プロジェクトA】ECサイトリニューアル
- リードエンジニアとして5名のチームをマネジメント
- マイクロサービスアーキテクチャへの移行を主導
- レスポンスタイムを50%改善、売上20%向上

【プロジェクトB】機械学習モデル開発
- レコメンデーションシステムの開発
- 精度90%を達成、A/Bテストで効果を検証
- 本番環境へのデプロイと運用を担当
"""


def test_without_rag():
    """RAGなしでテスト"""
    print("=" * 60)
    print("スモークテスト: RAGなし")
    print("=" * 60)
    
    try:
        # F1: 求人要件抽出
        print("\n[F1] 求人要件抽出...")
        requirements = extract_requirements(SAMPLE_JOB_TEXT, {})
        print(f"✅ F1完了: {len(requirements)}件の要件を抽出")
        
        # F2: 根拠抽出（RAGなし）
        print("\n[F2] 根拠抽出（RAGなし）...")
        evidence_map = extract_evidence(SAMPLE_RESUME_TEXT, requirements, {})
        print(f"✅ F2完了: {len(evidence_map)}件の根拠を分析")
        
        # F3: スコア計算
        print("\n[F3] スコア計算...")
        score_total, score_must, score_want, matched, gaps, summary = calculate_scores(
            requirements, evidence_map
        )
        print(f"✅ F3完了: 総合スコア {score_total}点 (Must: {score_must}, Want: {score_want})")
        
        # F4: 改善案生成
        print("\n[F4] 改善案生成...")
        improvements = generate_improvements(
            SAMPLE_JOB_TEXT, SAMPLE_RESUME_TEXT, requirements, matched, gaps, {}
        )
        print(f"✅ F4完了: {len(improvements.action_items)}件の行動計画を生成")
        
        # F5: 面接Q&A生成
        print("\n[F5] 面接Q&A生成...")
        interview_qas = generate_interview_qa(
            SAMPLE_JOB_TEXT, SAMPLE_RESUME_TEXT, matched, gaps, summary, {}
        )
        print(f"✅ F5完了: {len(interview_qas.qa_list)}件のQ&Aを生成")
        
        # F6: 品質評価（オプション、失敗しても続行）
        print("\n[F6] 品質評価...")
        try:
            quality_eval = evaluate_quality(
                SAMPLE_JOB_TEXT, SAMPLE_RESUME_TEXT, matched, gaps, improvements, interview_qas, {}
            )
            print(f"✅ F6完了: 総合品質スコア {quality_eval.overall_score:.1f}点")
        except Exception as e:
            print(f"⚠️  F6スキップ: {e}")
        
        print("\n" + "=" * 60)
        print("✅ 全テスト完了（RAGなし）")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_rag():
    """RAGありでテスト"""
    print("\n" + "=" * 60)
    print("スモークテスト: RAGあり")
    print("=" * 60)
    
    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEYが設定されていません。RAGテストをスキップします。")
        return True
    
    try:
        # F1: 求人要件抽出
        print("\n[F1] 求人要件抽出...")
        requirements = extract_requirements(SAMPLE_JOB_TEXT, {})
        print(f"✅ F1完了: {len(requirements)}件の要件を抽出")
        
        # F2: 根拠抽出（RAGあり）
        print("\n[F2] 根拠抽出（RAGあり）...")
        options_with_rag = {"achievement_notes": SAMPLE_ACHIEVEMENT_NOTES}
        evidence_map = extract_evidence(SAMPLE_RESUME_TEXT, requirements, options_with_rag)
        print(f"✅ F2完了: {len(evidence_map)}件の根拠を分析")
        
        # 引用の出どころを確認
        rag_quotes_count = 0
        resume_quotes_count = 0
        for ev in evidence_map.values():
            if ev.quote_sources:
                for source in ev.quote_sources:
                    if source == "rag":
                        rag_quotes_count += 1
                    elif source == "resume":
                        resume_quotes_count += 1
        print(f"  引用内訳: 職務経歴書={resume_quotes_count}件, RAG={rag_quotes_count}件")
        
        # F3: スコア計算
        print("\n[F3] スコア計算...")
        score_total, score_must, score_want, matched, gaps, summary = calculate_scores(
            requirements, evidence_map
        )
        print(f"✅ F3完了: 総合スコア {score_total}点 (Must: {score_must}, Want: {score_want})")
        
        # F4: 改善案生成
        print("\n[F4] 改善案生成...")
        improvements = generate_improvements(
            SAMPLE_JOB_TEXT, SAMPLE_RESUME_TEXT, requirements, matched, gaps, {}
        )
        print(f"✅ F4完了: {len(improvements.action_items)}件の行動計画を生成")
        
        # F5: 面接Q&A生成
        print("\n[F5] 面接Q&A生成...")
        interview_qas = generate_interview_qa(
            SAMPLE_JOB_TEXT, SAMPLE_RESUME_TEXT, matched, gaps, summary, {}
        )
        print(f"✅ F5完了: {len(interview_qas.qa_list)}件のQ&Aを生成")
        
        # F6: 品質評価（オプション、失敗しても続行）
        print("\n[F6] 品質評価...")
        try:
            quality_eval = evaluate_quality(
                SAMPLE_JOB_TEXT, SAMPLE_RESUME_TEXT, matched, gaps, improvements, interview_qas, {}
            )
            print(f"✅ F6完了: 総合品質スコア {quality_eval.overall_score:.1f}点")
        except Exception as e:
            print(f"⚠️  F6スキップ: {e}")
        
        print("\n" + "=" * 60)
        print("✅ 全テスト完了（RAGあり）")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン実行"""
    print("\n" + "=" * 60)
    print("AI応募適合度チェッカー - スモークテスト")
    print("=" * 60)
    
    # RAGなしテスト
    result1 = test_without_rag()
    
    # RAGありテスト
    result2 = test_with_rag()
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    print(f"RAGなしテスト: {'✅ 成功' if result1 else '❌ 失敗'}")
    print(f"RAGありテスト: {'✅ 成功' if result2 else '❌ 失敗'}")
    
    if result1 and result2:
        print("\n✅ 全テスト成功")
        return 0
    else:
        print("\n❌ 一部のテストが失敗しました")
        return 1


if __name__ == "__main__":
    exit(main())



