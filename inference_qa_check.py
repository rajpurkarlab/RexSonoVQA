#!/usr/bin/env python3
"""
QA Sanity Check Script (Blind Guessing Test)
=============================================

This script tests model performance WITHOUT showing video clips.
Used to verify that questions require visual understanding and cannot
be answered purely from question text and medical knowledge.

Usage:
    python inference_qa_check.py --questions_dir benchmark_questions/mcq --output_dir blind_test_results/mcq
    python inference_qa_check.py --questions_dir benchmark_questions/free --output_dir blind_test_results/free
"""

import os
import json
import argparse
import time
import tempfile
import subprocess
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from google import genai
from google.genai import types


# ============================================================================
# CONFIGURATION
# ============================================================================

class QAInferenceConfig:
    """Configuration for QA inference."""
    
    # API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    MODEL_ID: str = "gemini-3-pro-preview"
    
    # Gemini 3 specific parameters
    THINKING_LEVEL: str = "low"  # low, medium, high
    MEDIA_RESOLUTION: str = "media_resolution_low"  # low/medium=70 tokens/frame
    TEMPERATURE: float = 1.0
    
    # Processing
    MAX_RETRIES: int = 3
    RETRY_DELAY_SEC: float = 2.0


# ============================================================================
# VIDEO UTILITIES
# ============================================================================

def extract_video_clip(video_path: str, start: float, end: float, output_path: str) -> str:
    """Extract video clip for a time window WITHOUT audio."""
    duration = end - start
    
    # Try with stream copy first (fastest, no re-encoding)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration), 
        "-c", "copy",
        "-an",
        "-avoid_negative_ts", "make_zero",
        "-loglevel", "error",
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # If copy fails, fallback to ultrafast re-encoding
    if result.returncode != 0:
        print(f"    ⚠️  Stream copy failed, re-encoding...", flush=True)
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration), 
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-an",
            "-loglevel", "error",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    
    return output_path


def find_video_file(video_dir: str, question_filename: str) -> Optional[str]:
    """
    Find the corresponding video file for a question json file.
    POCUS_of_the_Abdominal_Aorta.json -> POCUS_of_the_Abdominal_Aorta.mp4
    """
    # Remove legacy prefixes and '.json' suffix
    base_name = question_filename.replace("qa_", "").replace(".json", "")
    
    # Expect .mp4 inputs
    video_path = os.path.join(video_dir, base_name + ".mp4")
    if os.path.exists(video_path):
        return video_path
    
    return None


# ============================================================================
# GEMINI INFERENCE
# ============================================================================

class QAInference:
    """Handles VLM inference for QA evaluation."""
    
    def __init__(self, config: QAInferenceConfig = None):
        self.config = config or QAInferenceConfig()
        
        if not self.config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Use v1alpha API version for Gemini 3 models
        self.client = genai.Client(
            api_key=self.config.GEMINI_API_KEY,
            http_options={'api_version': 'v1alpha'}
        )
    
    def create_system_prompt(self) -> str:
        """Create system prompt for QA task (NO VIDEO - sanity check mode)."""
        return """You are an expert ultrasound clinician being asked questions about ultrasound examinations.

IMPORTANT: You are NOT being shown any video. You must answer based ONLY on the question text itself and your general medical knowledge.

Key Guidelines:
- You have NO video to reference - answer based purely on the question text
- Use your medical knowledge to give the most likely correct answer
- For multiple choice questions (MCQ): Start your response with "Answer: X" where X is the correct letter (A, B, C, or D), then provide a brief explanation.
- For free response questions: Provide a concise, accurate answer in 1-3 sentences based on standard ultrasound practice
- If the question seems to require specific visual information you don't have, make your best educated guess

Your response should reflect what a knowledgeable clinician would answer without seeing the actual video."""

    def ask_question(self, video_path: str, question: str, question_prefix: str) -> Dict[str, Any]:
        """
        Ask a single question with a video clip in a new conversation.
        """
        result = {
            "model_prediction": "",
            "raw_response": "",
            "latency_ms": 0.0,
            "success": False,
            "error_message": ""
        }
        
        start_time = time.time()
        
        try:
            print("  📡 Creating chat session...", flush=True)
            # Create a new chat session for this question
            system_prompt = self.create_system_prompt()
            
            chat = self.client.chats.create(
                model=self.config.MODEL_ID,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.config.TEMPERATURE,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=self.config.THINKING_LEVEL
                    )
                )
            )
            
            # SANITY CHECK MODE: No video, only question text
            print("Sanity check mode: NOT loading video", flush=True)
            
            # Remove [MCQ] or [FREE] prefix from question when sending to model
            clean_question = question.replace("[MCQ] ", "").replace("[FREE] ", "")
            
            print("Sending to Gemini API (text only)...", flush=True)
            # Send question only, no video
            response = chat.send_message([clean_question])
            print("Response received", flush=True)
            result["raw_response"] = response.text
            
            # Parse response based on question type
            response_text = result["raw_response"].strip()
            
            if question_prefix == "[MCQ]":
                # Enhanced extraction logic for MCQ
                # Look for "Answer: X" pattern
                match = re.search(r'Answer:\s*\*?([A-D])\b', response_text, re.IGNORECASE)
                if match:
                    result["model_prediction"] = match.group(1).upper()
                else:
                    # Fallback: Look for "A.", "B)", etc. at start of lines
                    match_line = re.search(r'^\s*\*?([A-D])[\.\)]', response_text, re.MULTILINE)
                    if match_line:
                         result["model_prediction"] = match_line.group(1).upper()
                    else:
                        # Last resort: first letter occurrence (risky but better than nothing if desperate)
                        # Actually, keeping it empty or trying to find "The answer is X"
                         match_embed = re.search(r'(?:answer|option) is\s*\*?([A-D])\b', response_text, re.IGNORECASE)
                         if match_embed:
                             result["model_prediction"] = match_embed.group(1).upper()
                         else:
                             # Old fallback - very naive
                             if response_text and response_text[0].upper() in "ABCD":
                                result["model_prediction"] = response_text[0].upper()
            else:
                # For free response, use the full response
                result["model_prediction"] = response_text
            
            result["success"] = True
            
        except Exception as e:
            result["error_message"] = str(e)
            result["success"] = False
        
        finally:
            result["latency_ms"] = (time.time() - start_time) * 1000
        
        return result


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def shuffle_mcq(question_text: str, correct_original_letter: str) -> tuple[str, str]:
    """
    Shuffles MCQ options and updates the correct answer letter.
    Returns (new_question_text, new_correct_letter).
    """
    lines = question_text.strip().split('\n')
    body_lines = []
    options = {}
    
    # Regex for "A. " "B. " etc.
    option_pattern = re.compile(r'^([A-D])\.\s+(.*)')
    
    parsing_options = False
    
    for line in lines:
        stripped = line.strip()
        match = option_pattern.match(stripped)
        if match:
            parsing_options = True
            options[match.group(1)] = match.group(2)
        else:
            # If we haven't started seeing options, it is body.
            # If we HAVE seen options, and this line doesn't match "A. ", 
            # it might be a continuation of the previous option? 
            # For simplicity, assuming single-line options as per generated format.
            if not parsing_options:
                body_lines.append(line)
    
    # Needs to have 4 options to strictly proceed
    if len(options) != 4:
        return question_text, correct_original_letter
        
    # Get the text of the correct answer
    correct_text = options.get(correct_original_letter.upper())
    if not correct_text:
        return question_text, correct_original_letter
        
    # Shuffle the option texts
    option_texts = list(options.values())
    random.shuffle(option_texts)
    
    # Rebuild
    new_options_part = []
    new_correct_letter = ""
    letters = ['A', 'B', 'C', 'D']
    
    for i, text in enumerate(option_texts):
        letter = letters[i]
        new_options_part.append(f"{letter}. {text}")
        if text == correct_text:
            new_correct_letter = letter
            
    new_question = "\n".join(body_lines) + "\n" + "\n".join(new_options_part)
    return new_question, new_correct_letter


def process_question_file(
    question_path: str,
    video_dir: str,
    output_dir: str,
    config: QAInferenceConfig,
    verbose: bool = True
) -> None:
    """Process a single question file and generate predictions (no video - blind test)."""
    
    question_filename = os.path.basename(question_path)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Processing: {question_filename}")
        print(f"{'='*70}")
    
    # Load QA pairs
    with open(question_path, 'r', encoding='utf-8') as f:
        qa_items = json.load(f)
    
    if verbose:
        print(f"Loaded {len(qa_items)} QA pairs")
    
    # Find corresponding video (for metadata, not used in blind test)
    video_path = find_video_file(video_dir, question_filename)
    if not video_path:
        print(f"Warning: Could not find video file for {question_filename} (continuing anyway)")
    
    if verbose:
        print(f"BLIND TEST MODE: Video will NOT be shown to model")
        print(f"Model: {config.MODEL_ID}")
        print("-" * 70)
    
    # Initialize inference
    inference = QAInference(config)
    
    # Process each QA item
    results = []
    success_count = 0
    
    for idx, item in enumerate(qa_items):
        question = item["question"]
        time_start = item["time_start"]
        time_end = item["time_end"]
        
        # Determine question type prefix
        question_prefix = "[MCQ]" if question.startswith("[MCQ]") else "[FREE]"
        
        # Shuffle MCQ options
        if question_prefix == "[MCQ]" and "answer" in item:
            new_question, new_answer = shuffle_mcq(question, item["answer"])
            if new_question != question:
                item["original_question"] = question
                item["original_answer"] = item["answer"]
                item["question"] = new_question
                item["answer"] = new_answer
                question = new_question
        
        if verbose:
            print(f"\n[{idx+1}/{len(qa_items)}] {question_prefix} {time_start:.1f}s - {time_end:.1f}s")
            print(f"Q: {question[:80]}..." if len(question) > 80 else f"Q: {question}")
        
        # SANITY CHECK: Skip video extraction, send question only
        try:
            print(f"Sanity check: Skipping video clip extraction", flush=True)
            
            # Ask question with retries (no video)
            prediction_result = None
            for attempt in range(config.MAX_RETRIES):
                prediction_result = inference.ask_question(None, question, question_prefix)
                
                if prediction_result["success"]:
                    break
                
                if attempt < config.MAX_RETRIES - 1:
                    if verbose:
                        print(f"Retry {attempt + 1}/{config.MAX_RETRIES - 1}...")
                    time.sleep(config.RETRY_DELAY_SEC)
            
            # Create result entry
            result_item = item.copy()
            result_item["model_prediction"] = prediction_result["model_prediction"]
            result_item["inference_metadata"] = {
                "raw_response": prediction_result["raw_response"],
                "latency_ms": prediction_result["latency_ms"],
                "success": prediction_result["success"],
                "error_message": prediction_result["error_message"]
            }
            
            results.append(result_item)
            
            if prediction_result["success"]:
                success_count += 1
                if verbose:
                    pred = prediction_result["model_prediction"]
                    if len(pred) > 100:
                        pred = pred[:100] + "..."
                    print(f"A: {pred}")
                    print(f"⏱️  {prediction_result['latency_ms']:.0f}ms")
            else:
                if verbose:
                    print(f"FAILED: {prediction_result['error_message']}")
        
        finally:
            # No cleanup needed in sanity check mode (no temp files created)
            pass
    
    # Save results
    base_name = question_filename.replace("qa_", "")  # Handle legacy files
    output_filename = "pred_blind_" + base_name
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Completed: {success_count}/{len(qa_items)} successful")
        print(f"Saved to: {output_filename}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Run blind guessing test (no video) on QA benchmark")
    parser.add_argument("--question_file", type=str, help="Single question JSON file to process")
    parser.add_argument("--questions_dir", type=str, default="benchmark_questions/mcq", help="Directory containing question files")
    parser.add_argument("--video_dir", type=str, default="videos_all", help="Directory containing video files (for reference only)")
    parser.add_argument("--output_dir", type=str, default="blind_test_results", help="Output directory for blind test predictions")
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview", help="Model ID")
    parser.add_argument("--thinking_level", type=str, default="low", choices=["low", "medium", "high"])
    parser.add_argument("--glob", type=str, default="*.json", help="Glob pattern for question files")
    parser.add_argument("--parallel", type=int, default=5, help="Number of files to process concurrently")
    
    args = parser.parse_args()
    
    # Setup config
    config = QAInferenceConfig()
    config.MODEL_ID = args.model
    config.THINKING_LEVEL = args.thinking_level
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process files
    if args.question_file:
        # Single file mode
        process_question_file(args.question_file, args.video_dir, args.output_dir, config)
    else:
        # Batch mode
        questions_dir = Path(args.questions_dir)
        question_files = sorted(questions_dir.glob(args.glob))
        
        if not question_files:
            print(f"No question files found matching {args.glob} in {questions_dir}")
            return
        
        # Filter out files that have already been processed
        pending_files = []
        for question_file in question_files:
            base_name = question_file.name.replace("qa_", "")  # Handle legacy files
            output_filename = "pred_blind_" + base_name
            output_path = os.path.join(args.output_dir, output_filename)
            if os.path.exists(output_path):
                print(f"⏭️  Skipping {question_file.name} (already processed: {output_filename})")
            else:
                pending_files.append(question_file)
        
        if not pending_files:
            print(f"\nAll {len(question_files)} files already processed!")
            return
        
        print(f"\nStarting BLIND TEST on {len(pending_files)} files ({len(question_files) - len(pending_files)} skipped)...")
        print(f"   Videos will NOT be shown to model")
        print(f"   Using {args.parallel} parallel workers\n")

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            future_to_file = {
                executor.submit(process_question_file, str(question_file), args.video_dir, args.output_dir, config): question_file
                for question_file in pending_files
            }
            for future in as_completed(future_to_file):
                future.result()
        
        print(f"\nBlind test batch processing complete!")


if __name__ == "__main__":
    main()
