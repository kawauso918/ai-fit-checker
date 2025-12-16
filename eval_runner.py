"""
自動評価ランナー
eval/job*.txt と eval/resume.txt を読み、F1〜F4を実行して結果JSONを保存
"""
import os
import json
import glob
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from f1_extract_requirements import extract_requirements
from f2_extract_evidence import extract_evidence
from f3_score import calculate_scores
from f4_generate_improvements import generate_improvements
from llm_judge import evaluate_with_llm_judge, summarize_judge_results
from models import Requirement, Evidence, RequirementWithEvidence, Gap, Improvements


def run_evaluation(
    job_file: str,
    resume_file: str,
    output_file: str,
    options: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    1つの求人票に対して評価を実行
    
    Args:
        job_file: 求人票ファイルのパス
        resume_file: 職務経歴書ファイルのパス
        output_file: 出力JSONファイルのパス
        options: オプション辞書（LLM設定など）
        
    Returns:
        Dict[str, Any]: 評価結果
    """
    print(f"\n{'='*60}")
    print(f"評価実行: {os.path.basename(job_file)}")
    print(f"{'='*60}")
    
    # ファイル読み込み
    with open(job_file, 'r', encoding='utf-8') as f:
        job_text = f.read()
    
    with open(resume_file, 'r', encoding='utf-8') as f:
        resume_text = f.read()
    
    # オプションのデフォルト値
    if options is None:
        options = {
            "llm_provider": "openai",
            "model_name": None,
            "max_must": 10,
            "max_want": 10,
            "strict_mode": False,
            "verify_quotes": True,
            "max_gaps": 5
        }
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "job_file": os.path.basename(job_file),
        "resume_file": os.path.basename(resume_file),
        "options": options,
        "execution": {}
    }
    
    try:
        # F1: 求人要件抽出
        print("\n[F1] 求人要件抽出中...")
        requirements = extract_requirements(job_text, options)
        result["execution"]["f1"] = {
            "status": "success",
            "requirement_count": len(requirements),
            "requirements": [req.model_dump() for req in requirements]
        }
        print(f"✅ F1完了: {len(requirements)}件の要件を抽出")
        
        # F2: 根拠抽出
        print("\n[F2] 根拠抽出中...")
        evidence_map = extract_evidence(resume_text, requirements, options)
        result["execution"]["f2"] = {
            "status": "success",
            "evidence_count": len(evidence_map),
            "evidence": {req_id: ev.model_dump() for req_id, ev in evidence_map.items()}
        }
        print(f"✅ F2完了: {len(evidence_map)}件の根拠を分析")
        
        # F3: スコア計算
        print("\n[F3] スコア計算中...")
        score_total, score_must, score_want, matched, gaps, summary = calculate_scores(
            requirements, evidence_map
        )
        result["execution"]["f3"] = {
            "status": "success",
            "score_total": score_total,
            "score_must": score_must,
            "score_want": score_want,
            "matched_count": len(matched),
            "gap_count": len(gaps),
            "summary": summary,
            "matched": [
                {
                    "requirement": m.requirement.model_dump(),
                    "evidence": m.evidence.model_dump()
                }
                for m in matched
            ],
            "gaps": [
                {
                    "requirement": g.requirement.model_dump(),
                    "evidence": g.evidence.model_dump()
                }
                for g in gaps
            ]
        }
        print(f"✅ F3完了: 総合スコア {score_total}点 (Must: {score_must}, Want: {score_want})")
        
        # F4: 改善案生成
        print("\n[F4] 改善案生成中...")
        improvements = generate_improvements(
            job_text, resume_text, requirements, matched, gaps, options
        )
        result["execution"]["f4"] = {
            "status": "success",
            "improvements": improvements.model_dump()
        }
        print(f"✅ F4完了: {len(improvements.action_items)}件の行動計画を生成")
        
        # LLM-as-Judge: 評価（オプション）
        if options.get("enable_judge", False):
            print("\n[LLM-as-Judge] 評価実行中...")
            try:
                judge_output = evaluate_with_llm_judge(
                    job_text, resume_text, requirements, evidence_map,
                    matched, gaps, improvements, options
                )
                judge_summary = summarize_judge_results(judge_output)
                result["execution"]["judge"] = {
                    "status": "success",
                    "judge_output": judge_output.model_dump(),
                    "summary": judge_summary
                }
                print(f"✅ LLM-as-Judge完了: 総合スコア {judge_output.overall_score:.2f}")
            except Exception as judge_error:
                result["execution"]["judge"] = {
                    "status": "error",
                    "error": str(judge_error)
                }
                print(f"⚠️  LLM-as-Judgeエラー（無視）: {judge_error}")
        
        result["status"] = "success"
        print(f"\n✅ 評価完了: {os.path.basename(output_file)}")
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["error_type"] = type(e).__name__
        import traceback
        result["traceback"] = traceback.format_exc()
        print(f"\n❌ エラー発生: {e}")
        print(traceback.format_exc())
    
    # JSON保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return result


def run_all_evaluations(
    eval_dir: str = "eval",
    output_dir: str = "eval/outputs",
    options: Dict[str, Any] = None
):
    """
    全ての求人票に対して評価を実行
    
    Args:
        eval_dir: evalディレクトリのパス
        output_dir: 出力ディレクトリのパス
        options: オプション辞書
    """
    # 求人票ファイルを検索
    job_files = sorted(glob.glob(os.path.join(eval_dir, "job*.txt")))
    resume_file = os.path.join(eval_dir, "resume.txt")
    
    if not job_files:
        print(f"❌ 求人票ファイルが見つかりません: {eval_dir}/job*.txt")
        return
    
    if not os.path.exists(resume_file):
        print(f"❌ 職務経歴書ファイルが見つかりません: {resume_file}")
        return
    
    print(f"📋 評価対象: {len(job_files)}件の求人票")
    print(f"📄 職務経歴書: {resume_file}")
    print(f"💾 出力先: {output_dir}")
    
    results = []
    
    for job_file in job_files:
        # 出力ファイル名を決定（job1.txt -> job1.json）
        job_basename = os.path.basename(job_file)
        job_name = os.path.splitext(job_basename)[0]  # job1
        output_file = os.path.join(output_dir, f"{job_name}.json")
        
        # 評価実行
        result = run_evaluation(job_file, resume_file, output_file, options)
        results.append(result)
    
    # サマリー表示
    print(f"\n{'='*60}")
    print("📊 評価サマリー")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results if r.get("status") == "success")
    error_count = len(results) - success_count
    
    print(f"成功: {success_count}件 / エラー: {error_count}件")
    
    if success_count > 0:
        print("\nスコア一覧:")
        for result in results:
            if result.get("status") == "success":
                job_name = result.get("job_file", "unknown")
                f3 = result.get("execution", {}).get("f3", {})
                score_total = f3.get("score_total", "N/A")
                judge_info = ""
                if result.get("execution", {}).get("judge", {}).get("status") == "success":
                    judge_summary = result.get("execution", {}).get("judge", {}).get("summary", {})
                    judge_score = judge_summary.get("overall_score", "N/A")
                    if isinstance(judge_score, (int, float)):
                        judge_info = f" (Judge: {judge_score:.2f})"
                print(f"  {job_name}: {score_total}点{judge_info}")
    
    print(f"\n💾 結果は {output_dir} に保存されました")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="自動評価ランナー")
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="eval",
        help="evalディレクトリのパス（デフォルト: eval）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval/outputs",
        help="出力ディレクトリのパス（デフォルト: eval/outputs）"
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["openai", "anthropic"],
        default="openai",
        help="LLMプロバイダー（デフォルト: openai）"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="モデル名（デフォルト: プロバイダーのデフォルト）"
    )
    parser.add_argument(
        "--max-must",
        type=int,
        default=10,
        help="Must要件の最大件数（デフォルト: 10）"
    )
    parser.add_argument(
        "--max-want",
        type=int,
        default=10,
        help="Want要件の最大件数（デフォルト: 10）"
    )
    parser.add_argument(
        "--strict-mode",
        action="store_true",
        help="Strictモードを有効化"
    )
    parser.add_argument(
        "--no-verify-quotes",
        action="store_true",
        help="引用検証を無効化"
    )
    parser.add_argument(
        "--max-gaps",
        type=int,
        default=5,
        help="改善案生成時の最大ギャップ件数（デフォルト: 5）"
    )
    parser.add_argument(
        "--enable-judge",
        action="store_true",
        help="LLM-as-Judge評価を有効化"
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="LLM-as-JudgeのTemperature（デフォルト: 0.0）"
    )
    
    args = parser.parse_args()
    
    options = {
        "llm_provider": args.llm_provider,
        "model_name": args.model_name,
        "max_must": args.max_must,
        "max_want": args.max_want,
        "strict_mode": args.strict_mode,
        "verify_quotes": not args.no_verify_quotes,
        "max_gaps": args.max_gaps,
        "enable_judge": args.enable_judge,
        "judge_temperature": args.judge_temperature
    }
    
    run_all_evaluations(
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
        options=options
    )

