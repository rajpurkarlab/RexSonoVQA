# ReXSonoVQA: Ultrasound Video Benchmark

A comprehensive benchmark for evaluating vision-language models on ultrasound video understanding tasks.

## Overview

ReXSonoVQA provides a pipeline for:
1. **Building ground-truth annotations** from ultrasound videos
2. **Generating QA pairs** (MCQ and Free Response) from annotations
3. **Running model inference** against the benchmark
4. **Evaluating model performance** using LLM-as-judge

Video clips are streamed directly from YouTube during inference — **no local video files are needed**.

## Data Access

The file `video_metadata_new.json` provides human-curated metadata mapping each benchmark video to its YouTube source. This includes:
- **YouTube URL** for each video
- **Start/end timestamps** (seconds) locating the relevant segment within the YouTube video
- **Region of interest (ROI)** as proportional coordinates (0–1 fractions of the YouTube frame) to crop to the ultrasound content

No local video files are needed — clips are streamed directly from YouTube during inference.

## Directory Structure

```
ReXSonoVQA/
├── README.md
│
├── # ── Benchmark Data ──────────────────────────────────
├── benchmark_questions/        # Benchmark questions
│   ├── mcq/                    # MCQ questions (320 questions)
│   └── free/                   # Free Response questions (337 questions)
├── gt_all/                     # Ground-truth annotations
├── video_metadata_new.json     # YouTube metadata (URLs + timestamps + ROI)
│
├── # ── Benchmark Construction ──────────────────────────
├── build_benchmark.py          # Step 1: Build ground-truth from videos
├── generate_QA.py              # Step 2: Generate QA pairs from ground-truth
├── refine_MCQ.py               # Step 3: Refine MCQ distractors
├── inference_qa_check.py       # Step 4: Blind test for quality control
│
├── # ── Inference ────────────────────────────────────────
├── inference_qa_youtube.py     # Run inference via YouTube streaming
├── video_source.py             # YouTube video source utility
│
├── # ── Evaluation ──────────────────────────────────────
├── evaluate_results.py         # Score model predictions (LLM-as-judge)
│
├── # ── Baseline Results ────────────────────────────────
├── final_scores_all_MCQ/       # Baseline MCQ results
└── final_scores_all_free/      # Baseline Free Response results
```

## Quick Start: Testing a Model Against the Benchmark

Use `inference_qa_youtube.py` to stream clips directly from YouTube. This requires:
- `video_metadata_new.json` (included in the repo)
- `yt-dlp` and `ffmpeg` installed
- An API key for the chosen VLM backend

**Supported backends:** `gemini`, `qwen`, `seed`

```bash
# Gemini
export GEMINI_API_KEY="your-key"
python inference_qa_youtube.py \
    --backend gemini \
    --questions_dir benchmark_questions/mcq \
    --metadata_file video_metadata_new.json \
    --output_dir gemini_result/mcq

# Qwen
export DASHSCOPE_API_KEY="your-key"
python inference_qa_youtube.py \
    --backend qwen \
    --questions_dir benchmark_questions/mcq \
    --metadata_file video_metadata_new.json \
    --output_dir qwen_result/mcq

# Seed
export ARK_API_KEY="your-key"
python inference_qa_youtube.py \
    --backend seed \
    --questions_dir benchmark_questions/mcq \
    --metadata_file video_metadata_new.json \
    --output_dir seed_result/mcq
```

**YouTube mode options:**
- `--question_file FILE`: Process a single question file instead of the whole directory
- `--model MODEL_ID`: Override the default model ID for the chosen backend
- `--extend_start N` / `--extend_end N`: Extend each clip by N seconds at start/end
- `--parallel N`: Number of files to process in parallel (default: 3)

### Evaluate Results

Evaluation works the same regardless of which inference mode was used:

```bash
# Evaluate MCQ results
python evaluate_results.py \
    --pred_dir model_inference_results/mcq \
    --output_dir evaluation_scores/mcq

# Evaluate Free Response results
python evaluate_results.py \
    --pred_dir model_inference_results/free \
    --output_dir evaluation_scores/free
```

**Options:**
- `--force`: Force re-evaluation of all files
- `--parallel N`: Number of files to evaluate concurrently (default: 5)
- `--strict_judge`: Use stricter judging for vague/guessing answers

## YouTube Metadata Format

The `video_metadata_new.json` file maps each benchmark video to its YouTube source:

```json
{
  "1st_Trimester_Scan": {
    "youtube_url": "https://www.youtube.com/watch?v=PnoPvJXhanI",
    "start_time": 18.31,
    "end_time": 131.69,
    "roi": {
      "x_prop": 0.459016,
      "y_prop": 0.297917,
      "w_prop": 0.440281,
      "h_prop": 0.4375
    }
  }
}
```

| Field | Description |
|-------|-------------|
| `youtube_url` | Full YouTube URL for the source video |
| `start_time` | YouTube timestamp (seconds) where the benchmark content begins |
| `end_time` | YouTube timestamp (seconds) where the benchmark content ends |
| `roi` | Region of interest — proportional coordinates (`x_prop`, `y_prop`, `w_prop`, `h_prop`) as 0–1 fractions of the YouTube frame, cropping to the ultrasound content area |

QA timestamps (`time_start`, `time_end`) are **relative to the video content itself** — not to the full YouTube video. A question with `time_start=10.0` maps to YouTube time `start_time + 10.0`; the `video_source.py` utility handles this offset and ROI cropping automatically.

## Benchmark Data Format

The benchmark questions are stored in `benchmark_questions/mcq/` and `benchmark_questions/free/`. Each JSON file contains QA items with:

```json
{
  "question": "[MCQ] or [FREE] Question text...",
  "answer": "Ground truth answer",
  "groundtruth": "Details from original transcription",
  "question_type": "Type1_ActionGoalReasoning | Type2_... | Type3_...",
  "time_start": 0.0,
  "time_end": 10.5,
  "keep": true
}
```

- `time_start` / `time_end` are in seconds, relative to the **local video** (or equivalently, relative to `start_time` in the YouTube metadata).
- `keep` indicates whether the question passed blind-test quality control.

### Question Types

| Type | Description |
|------|-------------|
| `Type1_ActionGoalReasoning` | What maneuver is being performed and why (goal/target view) |
| `Type2_ArtifactResolutionOptimization` | Resolve artifacts/ambiguity: what changed and why |
| `Type3_ProcedureContextPlanning` | What step/phase, what's next, and why (protocol flow) |

## Building the Benchmark from Scratch

> **Note:** This section describes how the benchmark was originally constructed.
> You do **not** need to run these steps to use the benchmark — the questions in
> `benchmark_questions/` and the metadata in `video_metadata_new.json` are ready
> to use as-is.

### Step 1: Build Ground-Truth Annotations

Extract transcripts and generate structured ground-truth from videos:

```bash
python build_benchmark.py --video_dir videos_all --output_dir gt_all
```

This script:
- Extracts audio from videos
- Transcribes using WhisperX with word-level alignment
- Generates structured ground-truth events using Gemini 3 Pro

### Step 2: Generate QA Pairs

Generate MCQ and Free Response questions from ground-truth:

```bash
python generate_QA.py --gt_dir gt_all --output_dir qa_generated
```

This script:
- Reads ground-truth JSON files
- Generates diverse question types using GPT-5.2-pro
- Outputs structured QA pairs with timestamps

### Step 3: Refine MCQ Questions

Improve MCQ distractor quality:

```bash
python refine_MCQ.py --input_dir qa_generated --output_dir qa_refined
```

This script:
- Rewrites distractor options for MCQ questions
- Keeps the stem and correct answer unchanged
- Leaves Free Response questions untouched

### Step 4: Quality Control with Blind Test (Optional)

Run a blind test (no video) to verify questions require visual understanding:

```bash
# Test MCQ questions without showing video
python inference_qa_check.py --questions_dir qa_refined --output_dir blind_test_results

# Evaluate blind test results
python evaluate_results.py --pred_dir blind_test_results --output_dir blind_scores
```

Questions that can be answered correctly without video should be reviewed or removed, as they may be answerable from medical knowledge alone rather than requiring visual understanding.

**Manual QC with the `keep` field:**

Each evaluated question has a `keep` field (default: `true`). To exclude questions from final statistics:
1. Open the scored JSON files in `blind_scores/`
2. Set `"keep": false` for questions that should be excluded (e.g., answerable without video)
3. Re-run evaluation or use the filtered benchmark questions

The evaluation script automatically excludes items with `keep: false` from statistics.

### Step 5: Evaluate Performance

Score the model's predictions:

```bash
python evaluate_results.py \
    --pred_dir model_inference_results \
    --output_dir final_scores
```

## Evaluation Metrics

### MCQ Questions
- **Accuracy**: Exact match between predicted and correct answer letters

### Free Response Questions
- **Score (0-2)**: Evaluated by Gemini 3 Pro as LLM judge
  - `2` - Correct conclusion AND correct visual evidence
  - `1` - Partially correct (either conclusion or evidence is wrong)
  - `0` - Incorrect, irrelevant, or hallucinated

## Requirements

```
google-genai
openai
yt-dlp        # YouTube stream resolution
ffmpeg        # Clip extraction and ROI cropping
```

Install `yt-dlp` and `ffmpeg` via your package manager:
```bash
# macOS
brew install yt-dlp ffmpeg

# Ubuntu/Debian
pip install yt-dlp
sudo apt install ffmpeg
```
