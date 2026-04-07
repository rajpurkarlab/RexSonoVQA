#!/usr/bin/env python3
"""
QA Inference Script — YouTube Video Source (Multi-Backend)
==========================================================

This script replaces the local-video-file pipeline with one that accesses
content directly from YouTube using only URLs, timestamps, and ROI metadata.
**No video files are downloaded, stored, or redistributed.**

Supported backends
------------------
- **gemini**  — Google Gemini (native video upload via ``google-genai``)
- **gpt**     — OpenAI GPT (frame-based via Responses API)
- **qwen**    — Alibaba Qwen (native video via DashScope-compatible API)
- **seed**    — ByteDance Doubao Seed (native video via Ark API)

Usage
-----
    # Gemini
    python inference_qa_youtube.py \\
        --backend gemini \\
        --questions_dir benchmark_questions/mcq \\
        --metadata_file video_metadata.json \\
        --output_dir gemini_result/mcq

    # GPT
    python inference_qa_youtube.py \\
        --backend gpt \\
        --questions_dir benchmark_questions/free \\
        --metadata_file video_metadata.json \\
        --output_dir openai_result/free

    # Qwen
    python inference_qa_youtube.py \\
        --backend qwen \\
        --questions_dir benchmark_questions/mcq \\
        --metadata_file video_metadata.json \\
        --output_dir qwen_result/mcq

    # Seed
    python inference_qa_youtube.py \\
        --backend seed \\
        --questions_dir benchmark_questions/mcq \\
        --metadata_file video_metadata.json \\
        --output_dir seed_result/mcq
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from video_source import YouTubeVideoSource


# ============================================================================
# CONFIGURATION
# ============================================================================

class InferenceConfig:
    """Unified configuration for all backends."""

    # Backend selection (gemini | gpt | qwen | seed)
    BACKEND: str = "gemini"

    # ---- Gemini ----
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL_ID: str = "gemini-3-pro-preview"
    GEMINI_THINKING: str = "low"
    GEMINI_MEDIA_RES: str = "media_resolution_low"
    GEMINI_TEMPERATURE: float = 1.0

    # ---- OpenAI / GPT ----
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GPT_MODEL_ID: str = "gpt-5.4-2026-03-05"

    # ---- Qwen ----
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    QWEN_BASE_URL: str = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
    QWEN_MODEL_ID: str = "qwen3.5-397b-a17b"
    QWEN_ENABLE_THINKING: bool = True

    # ---- Seed ----
    ARK_API_KEY: str = os.getenv("ARK_API_KEY", "")
    ARK_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/v3"
    SEED_MODEL_ID: str = "doubao-seed-2-0-pro-260215"

    # ---- Common ----
    MAX_RETRIES: int = 3
    RETRY_DELAY_SEC: float = 2.0
    EXTEND_START_SEC: float = 0.0
    EXTEND_END_SEC: float = 0.0

    @property
    def active_model_id(self) -> str:
        return {
            "gemini": self.GEMINI_MODEL_ID,
            "gpt": self.GPT_MODEL_ID,
            "qwen": self.QWEN_MODEL_ID,
            "seed": self.SEED_MODEL_ID,
        }[self.BACKEND]


# ============================================================================
# HELPER UTILITIES
# ============================================================================

def encode_video_to_base64(video_path: str) -> str:
    """Read a video file and return its base64-encoded string."""
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_frames_from_clip(video_path: str, fps: float = 5.0) -> List[str]:
    """Extract frames from a video clip at the given FPS.
    Returns a list of base64-encoded JPEG strings (in chronological order).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_pattern = os.path.join(tmpdir, "frame_%05d.jpg")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"fps={fps}",
            "-q:v", "2",
            "-loglevel", "error",
            frame_pattern,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Frame extraction failed: {result.stderr}")
        frame_files = sorted(
            f for f in os.listdir(tmpdir)
            if f.startswith("frame_") and f.endswith(".jpg")
        )
        frames_b64 = []
        for fname in frame_files:
            with open(os.path.join(tmpdir, fname), "rb") as fh:
                frames_b64.append(base64.b64encode(fh.read()).decode("utf-8"))
    return frames_b64


def subsample_frames(frames: List[str], factor: int) -> List[str]:
    """Uniformly subsample frames by *factor*, always keeping first & last."""
    if len(frames) <= 2 or factor <= 1:
        return frames
    n = len(frames)
    target_count = max(2, n // factor)
    indices = sorted(set(
        [0]
        + [round(i * (n - 1) / (target_count - 1)) for i in range(target_count)]
        + [n - 1]
    ))
    return [frames[i] for i in indices]


def downsample_video(input_path: str, output_path: str,
                     scale_divisor: int = 2, crf: int = 28) -> str:
    """Re-encode a video at lower resolution / quality to reduce size."""
    scale_filter = (
        f"scale=iw/{scale_divisor}:ih/{scale_divisor}"
        f":force_original_aspect_ratio=decrease,"
        f"pad=ceil(iw/2)*2:ceil(ih/2)*2"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", scale_filter,
        "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
        "-an", "-loglevel", "error",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Video downsample failed: {result.stderr}")
    return output_path


SYSTEM_PROMPT_VIDEO = """You are an expert ultrasound clinician evaluating video clips from ultrasound examinations.

You will be shown an ultrasound video clip. The video contains ultrasound examination footage with the audio muted. You must answer based ONLY on what you observe visually in the video.

Key Guidelines:
- Watch the video carefully, observing probe movements, image changes, and anatomical structures
- Answer based solely on visual evidence from the video
- For multiple choice questions (MCQ): Start your response with "Answer: X" where X is the correct letter (A, B, C, or D), then provide a brief explanation.
- For free response questions: Provide a concise, accurate answer in 1-3 sentences
- If you cannot determine the answer from the video, state that clearly

Your response should be direct and factual, based on what you observe in the video."""


SYSTEM_PROMPT_FRAMES = """You are an expert ultrasound clinician evaluating video clips from ultrasound examinations.

You will be shown a sequence of frames extracted from an ultrasound video clip (at 5 FPS, in chronological order). The video contains ultrasound examination footage with the audio muted. You must answer based ONLY on what you observe visually in these frames.

Key Guidelines:
- Examine the frames carefully in order, observing probe movements, image changes, and anatomical structures
- Answer based solely on visual evidence from the frames
- For multiple choice questions (MCQ): Start your response with "Answer: X" where X is the correct letter (A, B, C, or D), then provide a brief explanation.
- For free response questions: Provide a concise, accurate answer in 1-3 sentences
- If you cannot determine the answer from the frames, state that clearly

Your response should be direct and factual, based on what you observe in the frames."""


# ============================================================================
# MCQ ANSWER EXTRACTION
# ============================================================================

def extract_mcq_answer(response_text: str) -> str:
    """Parse an MCQ answer letter from model response text."""
    # 1. "Answer: X"
    m = re.search(r'Answer:\s*\*?([A-D])\b', response_text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 2. "A." / "B)" at start of a line
    m = re.search(r'^\s*\*?([A-D])[\.\)]', response_text, re.MULTILINE)
    if m:
        return m.group(1).upper()
    # 3. "The answer is X"
    m = re.search(r'(?:answer|option) is\s*\*?([A-D])\b', response_text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 4. Naïve first-char
    if response_text and response_text[0].upper() in "ABCD":
        return response_text[0].upper()
    return ""


# ============================================================================
# BACKEND IMPLEMENTATIONS
# ============================================================================

class _BackendBase:
    """Abstract interface every backend must implement."""

    def ask(self, clip_path: str, question: str) -> str:
        """Send clip + question to the model and return the raw response text."""
        raise NotImplementedError


# ---- Gemini -----------------------------------------------------------------

class GeminiBackend(_BackendBase):
    def __init__(self, cfg: InferenceConfig):
        from google import genai
        from google.genai import types
        self._types = types
        if not cfg.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set")
        self.client = genai.Client(
            api_key=cfg.GEMINI_API_KEY,
            http_options={"api_version": "v1alpha"},
        )
        self.model = cfg.GEMINI_MODEL_ID
        self.cfg = cfg

    def ask(self, clip_path: str, question: str) -> str:
        types = self._types
        chat = self.client.chats.create(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_VIDEO,
                temperature=self.cfg.GEMINI_TEMPERATURE,
                thinking_config=types.ThinkingConfig(
                    thinking_level=self.cfg.GEMINI_THINKING,
                ),
            ),
        )
        with open(clip_path, "rb") as f:
            video_bytes = f.read()
        video_part = types.Part(
            inline_data=types.Blob(mime_type="video/mp4", data=video_bytes),
            media_resolution={"level": self.cfg.GEMINI_MEDIA_RES},
        )
        response = chat.send_message([video_part, question])
        return response.text


# ---- GPT (frame-based) -----------------------------------------------------

class GPTBackend(_BackendBase):
    def __init__(self, cfg: InferenceConfig):
        from openai import OpenAI
        if not cfg.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=cfg.OPENAI_API_KEY)
        self.model = cfg.GPT_MODEL_ID

    def ask(self, clip_path: str, question: str) -> str:
        all_frames = extract_frames_from_clip(clip_path, fps=5.0)
        if not all_frames:
            raise RuntimeError("No frames extracted from clip")

        subsample_factors = [1, 2, 4, 8]
        last_err = None
        for factor in subsample_factors:
            frames = subsample_frames(all_frames, factor)
            try:
                if factor > 1:
                    print(f"  🔄 Retrying with {len(frames)} frames "
                          f"(subsampled {factor}x)...", flush=True)
                else:
                    print(f"  📡 Sending {len(frames)} frames to GPT...",
                          flush=True)
                return self._send(frames, question)
            except Exception as e:
                last_err = e
                if "context_length_exceeded" in str(e).lower() or \
                   "context window" in str(e).lower():
                    continue
                raise
        raise last_err  # type: ignore[misc]

    def _send(self, frames_b64: List[str], question: str) -> str:
        user_content: list = []
        for fb in frames_b64:
            user_content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{fb}",
            })
        user_content.append({
            "type": "input_text",
            "text": (f"The above {len(frames_b64)} images are sequential "
                     f"frames from an ultrasound video clip.\n\n{question}"),
        })
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT_FRAMES},
                {"role": "user", "content": user_content},
            ],
        )
        return resp.output_text


# ---- Qwen ------------------------------------------------------------------

class QwenBackend(_BackendBase):
    DOWNSAMPLE_LEVELS = [None, (2, 28), (3, 32), (4, 35)]

    def __init__(self, cfg: InferenceConfig):
        from openai import OpenAI
        if not cfg.DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY not set")
        self.client = OpenAI(
            api_key=cfg.DASHSCOPE_API_KEY,
            base_url=cfg.QWEN_BASE_URL,
        )
        self.model = cfg.QWEN_MODEL_ID
        self.thinking = cfg.QWEN_ENABLE_THINKING

    def ask(self, clip_path: str, question: str) -> str:
        last_err = None
        ds_path = None
        try:
            for attempt, level in enumerate(self.DOWNSAMPLE_LEVELS):
                try:
                    if level is None:
                        b64 = encode_video_to_base64(clip_path)
                    else:
                        sd, crf = level
                        print(f"  🔄 Downsampling (1/{sd}, CRF {crf})...",
                              flush=True)
                        if ds_path and os.path.exists(ds_path):
                            os.remove(ds_path)
                        tmp = tempfile.NamedTemporaryFile(suffix=".mp4",
                                                         delete=False)
                        ds_path = tmp.name
                        tmp.close()
                        downsample_video(clip_path, ds_path, sd, crf)
                        b64 = encode_video_to_base64(ds_path)

                    mb = len(b64) * 3 / 4 / (1024 * 1024)
                    print(f"  📦 Video ~{mb:.1f} MB", flush=True)
                    return self._send(b64, question)

                except Exception as e:
                    last_err = e
                    is_size = any(
                        kw in str(e).lower() for kw in [
                            "max_string_length", "string value length",
                            "too large", "exceeds the maximum",
                            "payload too large", "content_length_exceeded",
                            "context_length_exceeded",
                        ]
                    )
                    if is_size and attempt < len(self.DOWNSAMPLE_LEVELS) - 1:
                        continue
                    raise
            raise last_err  # type: ignore[misc]
        finally:
            if ds_path and os.path.exists(ds_path):
                os.remove(ds_path)

    def _send(self, video_b64: str, question: str) -> str:
        user_content = [
            {"type": "video_url",
             "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
            {"type": "text", "text": question},
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_VIDEO},
                {"role": "user", "content": user_content},
            ],
            extra_body={"enable_thinking": self.thinking},
        )
        if resp.choices:
            return resp.choices[0].message.content
        raise RuntimeError("Empty response from Qwen API")


# ---- Seed -------------------------------------------------------------------

class SeedBackend(_BackendBase):
    def __init__(self, cfg: InferenceConfig):
        from openai import OpenAI
        if not cfg.ARK_API_KEY:
            raise ValueError("ARK_API_KEY not set")
        self.client = OpenAI(
            api_key=cfg.ARK_API_KEY,
            base_url=cfg.ARK_BASE_URL,
        )
        self.model = cfg.SEED_MODEL_ID

    def ask(self, clip_path: str, question: str) -> str:
        b64 = encode_video_to_base64(clip_path)
        mb = len(b64) * 3 / 4 / (1024 * 1024)
        print(f"  📦 Video ~{mb:.1f} MB", flush=True)
        user_content = [
            {"type": "input_video",
             "video_url": f"data:video/mp4;base64,{b64}"},
            {"type": "input_text", "text": question},
        ]
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT_VIDEO},
                {"role": "user", "content": user_content},
            ],
        )
        return resp.output_text


def make_backend(cfg: InferenceConfig) -> _BackendBase:
    """Factory: create the appropriate backend from config."""
    return {
        "gemini": GeminiBackend,
        "gpt": GPTBackend,
        "qwen": QwenBackend,
        "seed": SeedBackend,
    }[cfg.BACKEND](cfg)


# ============================================================================
# MCQ SHUFFLING
# ============================================================================

def shuffle_mcq(question_text: str, correct_original_letter: str) -> Tuple[str, str]:
    """Shuffle MCQ options and return (new_question, new_correct_letter)."""
    lines = question_text.strip().split("\n")
    body_lines: list[str] = []
    options: dict[str, str] = {}
    option_re = re.compile(r"^([A-D])\.\s+(.*)")
    parsing = False
    for line in lines:
        m = option_re.match(line.strip())
        if m:
            parsing = True
            options[m.group(1)] = m.group(2)
        elif not parsing:
            body_lines.append(line)

    if len(options) != 4:
        return question_text, correct_original_letter
    correct_text = options.get(correct_original_letter.upper())
    if not correct_text:
        return question_text, correct_original_letter

    texts = list(options.values())
    random.shuffle(texts)
    new_opts, new_letter = [], ""
    for i, t in enumerate(texts):
        letter = "ABCD"[i]
        new_opts.append(f"{letter}. {t}")
        if t == correct_text:
            new_letter = letter
    return "\n".join(body_lines) + "\n" + "\n".join(new_opts), new_letter


# ============================================================================
# PER-FILE PROCESSING
# ============================================================================

def process_question_file(
    question_path: str,
    video_source: YouTubeVideoSource,
    output_dir: str,
    config: InferenceConfig,
    verbose: bool = True,
) -> Tuple[str, int, int, str]:
    """Process a single question JSON file.

    Returns (filename, success_count, total_count, status).
    """
    question_filename = os.path.basename(question_path)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"📝 [START] Processing: {question_filename}")
        print(f"{'=' * 70}")

    # Load QA pairs
    with open(question_path, "r", encoding="utf-8") as f:
        qa_items = json.load(f)

    if not isinstance(qa_items, list):
        print(f"⏭ Skipping non-list file: {question_filename}")
        return (question_filename, 0, 0, "skipped: not a list")

    if verbose:
        print(f"Loaded {len(qa_items)} QA pairs")

    # Resolve video name
    video_name = YouTubeVideoSource.video_name_from_question_file(question_filename)
    if not video_source.has_video(video_name):
        print(f"❌ No metadata entry for video '{video_name}'")
        return (question_filename, 0, 0, "error: video not in metadata")

    # Query video duration (used for clamping extended clips)
    video_duration = video_source.get_video_duration(video_name)
    if verbose:
        print(f"Video: {video_name} (YouTube duration: {video_duration:.1f}s)")
        print(f"Backend: {config.BACKEND}  Model: {config.active_model_id}")
        if config.EXTEND_START_SEC or config.EXTEND_END_SEC:
            print(f"Extending clips by {config.EXTEND_START_SEC}s (start) "
                  f"and {config.EXTEND_END_SEC}s (end)")
        else:
            print("No time extension (exact clip boundaries)")
        print("-" * 70)

    # Create backend
    backend = make_backend(config)

    results: list[dict] = []
    success_count = 0

    for idx, item in enumerate(qa_items):
        question = item["question"]
        orig_start = item["time_start"]
        orig_end = item["time_end"]

        # Optional clip extension
        clip_start = max(0, orig_start - config.EXTEND_START_SEC)
        clip_end = min(video_duration, orig_end + config.EXTEND_END_SEC)

        question_prefix = "[MCQ]" if question.startswith("[MCQ]") else "[FREE]"

        # Shuffle MCQ options
        if question_prefix == "[MCQ]" and "answer" in item:
            new_q, new_a = shuffle_mcq(question, item["answer"])
            if new_q != question:
                item["original_question"] = question
                item["original_answer"] = item["answer"]
                item["question"] = new_q
                item["answer"] = new_a
                question = new_q

        if verbose:
            print(f"\n[{idx + 1}/{len(qa_items)}] {question_prefix}")
            print(f"  Clip: {clip_start:.1f}s – {clip_end:.1f}s")
            q_display = question[:80] + "..." if len(question) > 80 else question
            print(f"  Q: {q_display}")

        # Extract clip from YouTube
        clip_path: Optional[str] = None
        try:
            print(f"  🌐 Fetching clip from YouTube [{clip_start:.1f}s – {clip_end:.1f}s]...",
                  flush=True)
            clip_path = video_source.get_clip(video_name, clip_start, clip_end)
            clip_size = os.path.getsize(clip_path) / 1024
            print(f"  📦 Clip fetched ({clip_size:.0f} KB)", flush=True)

            # Remove prefix for model
            clean_question = question.replace("[MCQ] ", "").replace("[FREE] ", "")

            # Ask with retries
            pred_result: Dict[str, Any] = {
                "model_prediction": "",
                "raw_response": "",
                "latency_ms": 0.0,
                "success": False,
                "error_message": "",
            }
            t0 = time.time()

            for attempt in range(config.MAX_RETRIES):
                try:
                    print(f"  📡 Sending to {config.BACKEND} API...", flush=True)
                    raw = backend.ask(clip_path, clean_question)
                    pred_result["raw_response"] = raw
                    pred_result["success"] = True
                    print("  ✅ Response received", flush=True)
                    break
                except Exception as e:
                    pred_result["error_message"] = str(e)
                    if attempt < config.MAX_RETRIES - 1:
                        print(f"  ⚠️  Attempt {attempt + 1} failed, retrying...",
                              flush=True)
                        time.sleep(config.RETRY_DELAY_SEC)

            pred_result["latency_ms"] = (time.time() - t0) * 1000

            # Parse answer
            if pred_result["success"]:
                resp_text = pred_result["raw_response"].strip()
                if question_prefix == "[MCQ]":
                    pred_result["model_prediction"] = extract_mcq_answer(resp_text)
                else:
                    pred_result["model_prediction"] = resp_text

            # Build result entry (preserve original fields)
            result_item: dict = {}
            skip_keys = {
                "model_prediction", "inference_metadata", "eval_score",
                "eval_max_score", "judge_explanation", "judge_error_type",
                "duration",
            }
            for key in item:
                if key not in skip_keys:
                    result_item[key] = item[key]
            if "keep" in item:
                result_item["keep"] = item["keep"]

            result_item["original_time_start"] = orig_start
            result_item["original_time_end"] = orig_end
            result_item["model_prediction"] = pred_result["model_prediction"]
            result_item["inference_metadata"] = {
                "raw_response": pred_result["raw_response"],
                "latency_ms": pred_result["latency_ms"],
                "success": pred_result["success"],
                "error_message": pred_result["error_message"],
                "backend": config.BACKEND,
                "model_id": config.active_model_id,
                "video_source": "youtube",
                "extended_clip": bool(config.EXTEND_START_SEC or config.EXTEND_END_SEC),
            }
            if config.EXTEND_START_SEC or config.EXTEND_END_SEC:
                result_item["time_start_extended"] = clip_start
                result_item["time_end_extended"] = clip_end
                result_item["inference_metadata"]["extend_start_sec"] = config.EXTEND_START_SEC
                result_item["inference_metadata"]["extend_end_sec"] = config.EXTEND_END_SEC

            results.append(result_item)

            if pred_result["success"]:
                success_count += 1
                if verbose:
                    pred = pred_result["model_prediction"]
                    if len(pred) > 100:
                        pred = pred[:100] + "..."
                    print(f"  A: {pred}")
                    print(f"  ⏱ {pred_result['latency_ms']:.0f}ms")
            else:
                if verbose:
                    print(f"  ❌ FAILED: {pred_result['error_message']}")

        finally:
            if clip_path and os.path.exists(clip_path):
                os.remove(clip_path)

    # Save results
    base_name = question_filename.replace("score_pred_", "")
    output_filename = "pred_" + base_name
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"✅ [DONE] {question_filename}: {success_count}/{len(qa_items)} successful")
        print(f"   Saved to: {output_filename}")
        print(f"{'=' * 70}")

    return (question_filename, success_count, len(qa_items), "success")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run VLM inference on benchmark QA files using YouTube video source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument("--backend", type=str, required=True,
                        choices=["gemini", "gpt", "qwen", "seed"],
                        help="VLM backend to use")
    parser.add_argument("--questions_dir", type=str, required=True,
                        help="Directory with question JSON files")
    parser.add_argument("--metadata_file", type=str, required=True,
                        help="Path to video metadata JSON (YouTube URLs + ROI)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for prediction results")

    # Optional
    parser.add_argument("--question_file", type=str, default=None,
                        help="Process a single question file only")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model ID for the chosen backend")
    parser.add_argument("--glob", type=str, default="*.json",
                        help="Glob pattern for question files")
    parser.add_argument("--parallel", type=int, default=3,
                        help="Number of files to process in parallel")
    parser.add_argument("--extend_start", type=float, default=0.0,
                        help="Seconds to extend clip at start")
    parser.add_argument("--extend_end", type=float, default=0.0,
                        help="Seconds to extend clip at end")

    args = parser.parse_args()

    # Build config
    cfg = InferenceConfig()
    cfg.BACKEND = args.backend
    cfg.EXTEND_START_SEC = args.extend_start
    cfg.EXTEND_END_SEC = args.extend_end

    if args.model:
        attr = {
            "gemini": "GEMINI_MODEL_ID",
            "gpt": "GPT_MODEL_ID",
            "qwen": "QWEN_MODEL_ID",
            "seed": "SEED_MODEL_ID",
        }[args.backend]
        setattr(cfg, attr, args.model)

    # Initialize YouTube video source
    video_source = YouTubeVideoSource(args.metadata_file)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Backend : {cfg.BACKEND}")
    print(f"Model   : {cfg.active_model_id}")
    print(f"Metadata: {args.metadata_file} ({len(video_source.metadata)} videos)")
    print(f"Output  : {args.output_dir}")

    if args.question_file:
        # Single-file mode
        process_question_file(
            args.question_file, video_source, args.output_dir, cfg
        )
    else:
        # Batch mode
        questions_dir = Path(args.questions_dir)
        question_files = sorted(questions_dir.glob(args.glob))
        if not question_files:
            print(f"No files matching {args.glob} in {questions_dir}")
            return

        # Skip already-processed files
        pending = []
        for qf in question_files:
            base = qf.name.replace("score_pred_", "")
            out_name = "pred_" + base
            if os.path.exists(os.path.join(args.output_dir, out_name)):
                print(f"⏭ Skipping {qf.name} (already processed)")
            else:
                pending.append(qf)

        if not pending:
            print(f"\n✅ All {len(question_files)} files already processed!")
            return

        print(f"\nStarting inference on {len(pending)} files "
              f"({len(question_files) - len(pending)} skipped)...")
        print(f"   Parallel workers: {args.parallel}")

        results: list[tuple] = []
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    process_question_file,
                    str(qf), video_source, args.output_dir, cfg,
                ): qf
                for qf in pending
            }
            for future in as_completed(futures):
                res = future.result()
                if res:
                    results.append(res)

        # Summary
        print(f"\n{'=' * 70}")
        print("BATCH SUMMARY")
        print(f"{'=' * 70}")
        total_ok = sum(r[1] for r in results)
        total_q = sum(r[2] for r in results)
        ok_files = sum(1 for r in results if r[3] == "success")
        print(f"Files processed: {ok_files}/{len(pending)}")
        print(f"Questions: {total_ok}/{total_q} successful")
        print(f"\nPer-file results:")
        for fname, ok, total, status in sorted(results):
            icon = "✅" if status == "success" else "❌"
            print(f"   {icon} {fname}: {ok}/{total} ({status})")
        print(f"\n✅ Batch processing complete!")


if __name__ == "__main__":
    main()
