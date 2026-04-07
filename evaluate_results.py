#!/usr/bin/env python3
"""
QA Evaluation Script with LLM Judge
===================================

This script evaluates the inference results from `inference_qa.py`.
- MCQs are evaluated by exact match (Accuracy).
- Free Response questions are evaluated by Gemini 3 Pro as a judge (Score 0, 1, 2).
- Generates statistics by Question Type and Video Clip Duration.

Usage:
    python evaluate_results.py --pred_dir model_inference_results/ --output_dir final_scores/
"""

import os
import json
import argparse
import time
import statistics
import math
from pathlib import Path
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
from collections import defaultdict

from google import genai
from google.genai import types

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ============================================================================
# CONFIGURATION
# ============================================================================

class EvalConfig:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    JUDGE_MODEL_ID: str = "gemini-3-pro-preview"
    JUDGE_TEMPERATURE: float = 0.0
    STRICT_JUDGE: bool = True
    
# ============================================================================
# LLM JUDGE
# ============================================================================

class LLMJudge:
    def __init__(self, config: EvalConfig):
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set")
            
        self.client = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options={'api_version': 'v1alpha'}
        )
        self.model_id = config.JUDGE_MODEL_ID
        self.config = config

    def evaluate_free_response(self, question: str, ground_truth: str, prediction: str) -> Dict[str, Any]:
        """
        Evaluates a free response prediction against ground truth using LLM.
        Returns {"score": int, "explanation": str}
        """
        strict_note = """

STRICTNESS OVERRIDE:
- Penalize guessing or vague organ references.
- If the ground truth names a specific organ/structure/feature and the prediction only gives a generic organ reference (e.g., "an organ" or a different unspecified organ), treat this as incorrect visual evidence.
- If the prediction is uncertain or overly general (e.g., "likely an organ" without identification), do NOT award full credit. At most score=1, and classify the error as wrong_visual_evidence.
- Non-identification of the required organ/feature should be categorized as visual evidence error.
""".strip()

        system_prompt = """You are an impartial medical expert judge evaluating the quality of an AI assistant's response to a clinical ultrasound question.

    Compare the AI's Prediction against the Ground Truth Answer.

    Scoring Criteria:
    2 - Correct: The conclusion/answer matches the ground truth AND the cited visual evidence aligns with the ground truth. Minor phrasing differences are acceptable.
    1 - Partially Correct: Only one of the two is correct (either the conclusion is correct but the visual evidence is wrong, OR the visual evidence is correct but the conclusion is wrong).
    0 - Incorrect: The prediction is wrong, irrelevant, hallucinated, or contradicts the ground truth.

    If score = 1, classify the mistake type:
    - wrong_visual_evidence: conclusion is correct, but evidence/visual justification is incorrect or mismatched.
    - wrong_conclusion: evidence/visual cues are correct, but the conclusion/answer is wrong.

    If score = 0, classify the mistake type:
    - wrong_visual_evidence: evidence/visual justification is incorrect.
    - wrong_conclusion: conclusion/answer is incorrect.
    - both_fail: both evidence and conclusion are incorrect.

    Output exactly and ONLY valid JSON in this format:
    {"score": 0, "explanation": "Brief reasoning", "error_type": "none"}
        """

        if self.config.STRICT_JUDGE:
            system_prompt = system_prompt + "\n\n" + strict_note
        
        user_message = f"""
Question: {question}

Ground Truth: {ground_truth}

AI Prediction: {prediction}

Evaluate now.
"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    response_mime_type="application/json" 
                ),
                contents=[user_message]
            )
            
            # remove markdown fencing if present (though response_mime_type usually handles it)
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:-3].strip()
                
            result = json.loads(text)
            # Normalize missing fields
            if "error_type" not in result:
                result["error_type"] = "none"
            # Fill default error_type based on score if missing/none
            score = result.get("score", 0)
            if result["error_type"] == "none":
                if score == 1:
                    result["error_type"] = "wrong_conclusion"
                elif score == 0:
                    result["error_type"] = "both_fail"
            return result
            
        except Exception as e:
            print(f"Judge Error: {e}")
            return {"score": 0, "explanation": f"Evaluation Failed: {str(e)}", "error_type": "none"}

# ============================================================================
# SCORING LOGIC
# ============================================================================

def evaluate_file(file_path: str, judge: LLMJudge, output_dir: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    scored_items = []
    print(f"\nEvaluating {os.path.basename(file_path)} ({len(data)} items)...")
    
    for item in data:
        # Determine Type
        q_type = "MCQ" if item["question"].strip().startswith("[MCQ]") or item.get("question_type", "").startswith("Type") and len(item.get("answer", "")) == 1 else "FREE"
        if "MCQ" in item["question"]: q_type = "MCQ"
        if "FREE" in item["question"]: q_type = "FREE"
        
        gt = item["answer"]
        pred = item["model_prediction"]
        duration = item["time_end"] - item["time_start"]
        
        score_info = {}
        
        if q_type == "MCQ":
            # Clean up prediction just in case (e.g. "A." -> "A")
            # If pred is empty or not a letter, it's wrong.
            if pred and len(pred.strip()) > 0:
                clean_pred = pred.strip()[0].upper()
            else:
                clean_pred = ""
            
            clean_gt = gt.strip()[0].upper()
            
            # simple binary accuracy
            is_correct = (clean_pred == clean_gt)
            
            if not is_correct:
                print(f"  [MCQ Mismatch] Pred: '{clean_pred}' | GT: '{clean_gt}'")

            score_info = {
                "score": 1 if is_correct else 0, 
                "max_score": 1,
                "is_mcq": True
            }
        else:
            # Free Response
            eval_result = judge.evaluate_free_response(item["question"], gt, pred)
            score_info = {
                "score": eval_result.get("score", 0),
                "max_score": 2,
                "explanation": eval_result.get("explanation", ""),
                "error_type": eval_result.get("error_type", "none"),
                "is_mcq": False
            }
            print(f"  [FREE] Score: {score_info['score']}/2")

        # Save score back to item
        item["eval_score"] = score_info["score"]
        item["eval_max_score"] = score_info["max_score"]
        item["duration"] = duration
        if "keep" not in item:
            item["keep"] = True  # Default to keep; set to False during QC to exclude
        if "explanation" in score_info:
            item["judge_explanation"] = score_info["explanation"]
        if "error_type" in score_info:
            item["judge_error_type"] = score_info["error_type"]
            
        scored_items.append(item)

    # Save scored file
    out_name = "score_" + os.path.basename(file_path)
    with open(os.path.join(output_dir, out_name), 'w') as f:
        json.dump(scored_items, f, indent=2)
        
    return scored_items

# ============================================================================
# STATISTICS
# ============================================================================

def print_statistics(all_items: List[Dict], output_dir: str):
    total_mcq = 0
    correct_mcq = 0
    
    total_free = 0
    total_free_score = 0
    
    by_type = defaultdict(lambda: {"total": 0, "score": 0, "max_score": 0})
    by_type_free = defaultdict(lambda: {"total": 0, "score": 0})  # FREE only
    free_partial_errors = defaultdict(int)
    free_error_type_counts = defaultdict(int)
    
    # Duration bins: 0-5s, 5-10s, 10-20s, >20s
    duration_bins = {
        "0-5s": {"total": 0, "score_norm": 0.0},
        "5-10s": {"total": 0, "score_norm": 0.0},
        "10-20s": {"total": 0, "score_norm": 0.0},
        ">20s": {"total": 0, "score_norm": 0.0},
    }
    duration_bins_free = {
        "0-5s": {"total": 0, "score": 0},
        "5-10s": {"total": 0, "score": 0},
        "10-20s": {"total": 0, "score": 0},
        ">20s": {"total": 0, "score": 0},
    }
    
    def normalize_question_type(q_type: str) -> str:
        if q_type in {"Type4_DisambiguationSafetyDecision", "Type2_OptimizationTroubleshooting"}:
            return "Type2_ArtifactResolutionOptimization"
        return q_type

    free_items_for_plot: List[Dict] = []
    for item in all_items:
        # Only include items explicitly kept
        if item.get("keep", True) is not True:
            continue
        score = item["eval_score"]
        max_s = item["eval_max_score"]
        q_type_detailed = normalize_question_type(item.get("question_type", "Unknown"))
        duration = item["duration"]
        
        # Normalized score (0.0 to 1.0) for aggregation
        norm_score = score / max_s if max_s > 0 else 0
        
        # Determine if MCQ or FREE
        is_mcq = item.get("question", "").startswith("[MCQ]") or max_s == 1
        
        # Duration bin
        if duration < 5: bin_name = "0-5s"
        elif duration < 10: bin_name = "5-10s"
        elif duration < 20: bin_name = "10-20s"
        else: bin_name = ">20s"
        
        # Overall Stats
        if is_mcq:
            total_mcq += 1
            if score == 1: correct_mcq += 1
            # By Type (MCQ only)
            by_type[q_type_detailed]["total"] += 1
            by_type[q_type_detailed]["score"] += score
            by_type[q_type_detailed]["max_score"] += max_s

            # By Duration (MCQ only)
            duration_bins[bin_name]["total"] += 1
            duration_bins[bin_name]["score_norm"] += norm_score
        else:
            total_free += 1
            total_free_score += score
            free_items_for_plot.append(item)
            # FREE-only by type
            by_type_free[q_type_detailed]["total"] += 1
            by_type_free[q_type_detailed]["score"] += score
            # FREE-only by duration
            duration_bins_free[bin_name]["total"] += 1
            duration_bins_free[bin_name]["score"] += score
            # FREE-only partial error type
            if score == 1:
                err_type = item.get("judge_error_type", "unknown")
                free_partial_errors[err_type] += 1
            # FREE-only error type proportions (against all FREE questions)
            if score in (0, 1):
                if score == 0:
                    free_error_type_counts["wrong_both"] += 1
                else:
                    err_type_all = item.get("judge_error_type", "unknown")
                    if err_type_all == "wrong_conclusion":
                        free_error_type_counts["wrong_conclusion_only"] += 1
                    elif err_type_all == "wrong_visual_evidence":
                        free_error_type_counts["wrong_visual_evidence_only"] += 1
                    else:
                        free_error_type_counts["wrong_both"] += 1
            
        # (By Type/Duration for FREE handled above where applicable)

    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    print(f"\nOVERALL")
    if total_mcq > 0:
        print(f"MCQ Accuracy: {correct_mcq}/{total_mcq} ({correct_mcq/total_mcq*100:.1f}%)")
    if total_free > 0:
        # Scale 0-2 -> Divide by (count * 2)
        print(f"Free Response Avg Score: {total_free_score/total_free:.2f} / 2.0")
        print(f"Free Response Accuracy (Norm): {(total_free_score / (total_free * 2))*100:.1f}%")

    print(f"\nBY QUESTION TYPE (MCQ Only)")
    print(f"{'Type':<40} | {'Count':<5} | {'Accuracy':<8}")
    print("-" * 65)
    for t, data in sorted(by_type.items()):
        acc = (data["score"] / data["max_score"]) * 100 if data["max_score"] else 0
        print(f"{t:<40} | {data['total']:<5} | {acc:.1f}%")

    print(f"\nBY QUESTION TYPE (FREE Only)")
    print(f"{'Type':<40} | {'Count':<5} | {'Avg Score':<10}")
    print("-" * 65)
    for t, data in sorted(by_type_free.items()):
        avg = data["score"] / data["total"] if data["total"] > 0 else 0
        print(f"{t:<40} | {data['total']:<5} | {avg:.2f} / 2.0")

    if total_free > 0:
        print(f"\nFREE PARTIAL ERROR TYPES (Score=1)")
        print(f"{'Error Type':<24} | {'Count':<5}")
        print("-" * 33)
        if free_partial_errors:
            for err_type, count in sorted(free_partial_errors.items()):
                print(f"{err_type:<24} | {count:<5}")
        else:
            print("none                   | 0    ")

        print(f"\nFREE ERROR TYPE PROPORTIONS (All FREE)")
        print(f"{'Error Type':<28} | {'Count':<5} | {'Proportion':<10}")
        print("-" * 46)
        if total_free > 0:
            for err_type in ["wrong_conclusion_only", "wrong_visual_evidence_only", "wrong_both"]:
                count = free_error_type_counts.get(err_type, 0)
                proportion = (count / total_free) * 100 if total_free > 0 else 0
                print(f"{err_type:<28} | {count:<5} | {proportion:>6.1f}%")
        else:
            print(f"none{'':<24} | 0     |   0.0%")

    print(f"\nBY VIDEO DURATION (MCQ Only)")
    print(f"{'Duration':<10} | {'Count':<5} | {'Avg Norm Score':<8}")
    print("-" * 40)
    for bin_name in ["0-5s", "5-10s", "10-20s", ">20s"]:
        d = duration_bins[bin_name]
        avg = (d["score_norm"] / d["total"] * 100) if d["total"] > 0 else 0
        print(f"{bin_name:<10} | {d['total']:<5} | {avg:.1f}%")

    print(f"\nBY VIDEO DURATION (FREE Only)")
    print(f"{'Duration':<10} | {'Count':<5} | {'Avg Score':<10}")
    print("-" * 40)
    for bin_name in ["0-5s", "5-10s", "10-20s", ">20s"]:
        d = duration_bins_free[bin_name]
        avg = d["score"] / d["total"] if d["total"] > 0 else 0
        print(f"{bin_name:<10} | {d['total']:<5} | {avg:.2f} / 2.0")

    # Plot: video duration vs FREE response avg score (3s bins)
    if free_items_for_plot:
        max_duration = max(it.get("duration", 0) for it in free_items_for_plot)
        bin_size = 3
        bin_count = max(1, int(math.ceil(max_duration / bin_size)))
        bin_totals = [0.0] * bin_count
        bin_counts = [0] * bin_count
        for it in free_items_for_plot:
            dur = float(it.get("duration", 0))
            score = float(it.get("eval_score", 0))
            idx = min(int(dur // bin_size), bin_count - 1)
            bin_totals[idx] += score
            bin_counts[idx] += 1

        bin_edges = [i * bin_size for i in range(bin_count + 1)]
        x_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(bin_count)]
        avg_scores = [
            (bin_totals[i] / bin_counts[i]) if bin_counts[i] > 0 else 0
            for i in range(bin_count)
        ]

        out_path = Path(output_dir) / "free_response_avg_by_duration_3s.jpg"
        if plt is None:
            print("\nmatplotlib not available; skipping duration plot.")
        else:
            plt.figure(figsize=(max(8, bin_count * 0.35), 4))
            plt.plot(range(bin_count), avg_scores, marker="o")
            plt.xticks(range(bin_count), x_labels, rotation=45, ha="right")
            plt.ylim(0, 2)
            plt.ylabel("Avg FREE Score (0-2)")
            plt.xlabel("Duration (s) bins of 3s")
            plt.title("FREE Response Avg Score vs Video Duration")
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"\nSaved duration plot: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--force", action="store_true", help="Force re-evaluation of all files")
    parser.add_argument("--parallel", type=int, default=5, help="Number of result files to evaluate concurrently")
    parser.add_argument("--strict_judge", action="store_true", help="Use stricter judging against vague/guessing answers")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = EvalConfig()
    config.STRICT_JUDGE = args.strict_judge
    
    pred_files = glob.glob(os.path.join(args.pred_dir, "pred_*.json"))
    all_items = []
    
    pending_pred_files = []
    for pf in pred_files:
        # Check if already scored
        score_filename = "score_" + os.path.basename(pf)
        score_path = os.path.join(args.output_dir, score_filename)
        
        if os.path.exists(score_path) and not args.force:
            # Load existing scores
            print(f"Loading existing scores: {score_filename}")
            with open(score_path, 'r') as f:
                items = json.load(f)
            all_items.extend(items)
        else:
            pending_pred_files.append(pf)

    if pending_pred_files:
        print(f"\nEvaluating {len(pending_pred_files)} files concurrently (workers={args.parallel})...")

        def evaluate_with_new_judge(pf: str) -> List[Dict]:
            judge = LLMJudge(config)
            return evaluate_file(pf, judge, args.output_dir)

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            future_to_pf = {executor.submit(evaluate_with_new_judge, pf): pf for pf in pending_pred_files}
            for future in as_completed(future_to_pf):
                items = future.result()
                all_items.extend(items)
    
    # Also load any score files that don't have corresponding pred files (in case pred was deleted)
    existing_score_files = glob.glob(os.path.join(args.output_dir, "score_*.json"))
    # Score files are named "score_pred_X.json" so we need "score_" + basename(pred_file)
    loaded_score_names = {"score_" + os.path.basename(pf) for pf in pred_files}
    
    for sf in existing_score_files:
        if os.path.basename(sf) not in loaded_score_names:
            print(f"Loading orphan score file: {os.path.basename(sf)}")
            with open(sf, 'r') as f:
                items = json.load(f)
            all_items.extend(items)
        
    print_statistics(all_items, args.output_dir)
    
    # Save batch summary
    summary_path = os.path.join(args.output_dir, "batch_summary.json")
    summary = {
        "total_items": len(all_items),
        "files_processed": len(pred_files),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    main()
