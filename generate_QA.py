#!/usr/bin/env python3
"""
Generate Ultrasound video-only Q&A benchmark items from transcript-derived ground-truth events.

- Reads *.json ground-truth files from an input directory. Each file is expected to be a JSON list of events:
  [{"start": float, "end": float, "action": str, "interpretation": str}, ...]
- Calls OpenAI Responses API (default model: gpt-5.2-pro) with Structured Outputs (json_schema)
- Writes one QA json per input file to the output directory.

Output schema per item:
{
    "question": str,        # Prefix with [MCQ] or [FREE]
    "answer": str,          # For MCQ, include the correct option (e.g., "B) ...")
    "question_type": str,   # One of 3 types
    "groundtruth": str,     # Rephrased single entry combining action+interpretation
    "time_start": float,
    "time_end": float
}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI


# ----------------------------
# Configuration
# ----------------------------

QUESTION_TYPES = [
    "Type1_ActionGoalReasoning",                # what maneuver + why (goal/target view)
    "Type2_ArtifactResolutionOptimization",     # resolve artifacts/ambiguity: what changed + why
    "Type3_ProcedureContextPlanning",           # what step/phase + what next + why (protocol flow)
]

# Hard constraints to keep "video-only / muted / no OCR" honest:
BANNED_PHRASES_REGEX = re.compile(
    r"\b(audio|narrat|transcript|spoken|said|instruct(ed)? (the patient|them) to|on-screen text|read the text)\b",
    flags=re.IGNORECASE,
)

# If a ground-truth interpretation includes numeric readouts, don't ask for exact numbers.
NUMERIC_READOUT_REGEX = re.compile(r"\b\d+(\.\d+)?\s*(mm|cm|bpm|hz|khz)\b", flags=re.IGNORECASE)


@dataclass
class Event:
    start: float
    end: float
    action: str
    interpretation: str


def load_events(path: Path) -> List[Event]:
    print(f"  Loading events from {path.name}...")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path.name}: expected a JSON list, got {type(raw)}")

    events: List[Event] = []
    for i, obj in enumerate(raw):
        if not isinstance(obj, dict):
            continue
        if not all(k in obj for k in ("start", "end", "action", "interpretation")):
            continue
        try:
            events.append(
                Event(
                    start=float(obj["start"]),
                    end=float(obj["end"]),
                    action=str(obj["action"]).strip(),
                    interpretation=str(obj["interpretation"]).strip(),
                )
            )
        except Exception as e:
            raise ValueError(f"{path.name}: bad event at index {i}: {e}") from e

    # Sort just in case
    events.sort(key=lambda e: (e.start, e.end))
    print(f"  Loaded {len(events)} events")
    return events


def chunk_events(events: List[Event], max_chars: int = 45000) -> List[List[Event]]:
    """
    Avoid giant prompts. If serialized events exceed max_chars, split into chunks.
    Chunks overlap by 1 event to preserve adjacency context.
    """
    chunks: List[List[Event]] = []
    cur: List[Event] = []
    cur_len = 0

    def est_len(ev: Event) -> int:
        return len(ev.action) + len(ev.interpretation) + 50

    for ev in events:
        add_len = est_len(ev)
        if cur and (cur_len + add_len) > max_chars:
            chunks.append(cur)
            # overlap by 1 event
            cur = [cur[-1], ev]
            cur_len = est_len(cur[0]) + add_len
        else:
            cur.append(ev)
            cur_len += add_len

    if cur:
        chunks.append(cur)
    if len(chunks) > 1:
        print(f"  Split into {len(chunks)} chunks")
    return chunks


def build_system_prompt() -> str:
    return f"""
You are generating Q&A items for a benchmark that evaluates a Vision-Language Model (VLM) on ULTRASOUND DEMONSTRATION VIDEOS.

CRITICAL EVAL SETUP (what the VLM will see at test time):
- Video ONLY (muted). No audio.
- On-screen TEXT READING IS NOT ALLOWED (assume OCR is disallowed).
- Therefore, every question must be answerable from VISUAL cues only:
  - probe motion (slide/sweep/rock/rotate/tilt, directionality, marker orientation when visible),
  - patient positioning/motion if visually observable,
  - ultrasound image dynamics (structures entering/leaving frame, plane changes, motion, shadowing),
  - mode changes that are visually obvious (e.g., Color Doppler overlay appears/disappears),
  - imaging parameter changes if visible in the image effect (e.g., depth scale changes, focal zone marker moves, sector narrows).

GROUND TRUTH YOU WILL RECEIVE:
- A sequence of time-stamped events extracted from transcript, each with:
  - action (what operator did)
  - interpretation (why / what they were looking for)
These are NOT available to the VLM at test time; they are for YOU to craft correct Q/A.

THE MOST CRITICAL QUESTION CONSTRUCTION RULE:
Each question must be answerable only by watching the video. Do not include visual clues, descriptions, or contextual hints that would allow someone to answer or guess correctly without viewing the video. 
Do not contain specific visual details, structures, context, artifacts, or outcomes in the question itself.

BAD (includes visual clues): "When the operator sweeps inferiorly and identifies the bifurcation, what anatomical landmark is being visualized?"
GOOD (requires watching): "Based on the clip, what acquisition goal is being pursued during the continuous inferior sweep, and what key anatomic endpoint indicates success?"

BAD (describes what's visible): "As bowel gas obscures the vessel view and the operator applies firm pressure..."
GOOD (asks about action/goal): "The clip shows loss/obscuration of the target vessel. What optimization maneuver is performed, and what problem is it addressing to restore the view?"

Always phrase questions as:
- "Based on the video/clip, what..."
- "What [action/strategy/technique] is shown, and what is the goal/objective?"
- "This clip shows [general/non-specific observation]. What is being done and why?"

You must generate items in exactly 3 clinically-meaningful question types:

1) Type1_ActionGoalReasoning
   - Tests: action reasoning + goal inference.
   - Ask: what maneuver is being performed AND what imaging goal / target view it serves.
   - Phrase questions naturally and adaptively based on the content.
   - Do NOT describe the specific anatomical structures or visual details in the question.

2) Type2_ArtifactResolutionOptimization
   - Tests: overcoming artifacts or ambiguity + optimization/disambiguation logic.
   - Ask: what (probe maneuver, patient management, or knobology) has changed AND why it resolves an artifact or ambiguity / improves image quality.
   - IMPORTANT: Do not explicitly describe the artifact or ambiguity in the question.
   - Phrase questions naturally - you can reference general observations like "loss of view" or "poor quality" without describing specific details.

3) Type3_ProcedureContextPlanning
   - Tests: overall context understanding + next-step planning.
   - Ask: what phase/step the operator is in AND what the broader workflow objective or next logical step is.
   - Usually use TWO or more ADJACENT EVENTS to create sufficient context.
   - Vary your phrasing - ask about exam phases, workflow transitions, procedural objectives, or strategies as appropriate.
   - Do not describe the clip’s content or sequence and do not name specific anatomy or maneuvers in the question itself.


QUESTION FORMAT MIX:
- Produce a mix of MCQ and free response.
- For MCQ: 
  * Put exactly 4 options (A–D) inside the "question" string. Ensure only ONE is correct; distractors must be plausible.
  * Keep options generic and hypothesis-based when possible
  * Do NOT make options that give away visual details that should only be known from watching the video
  * IMPORTANT: The "answer" field must contain ONLY the letter of the correct option (e.g., "A" or "B" or "C" or "D")
- For Free response: 
  * The "question" must be answerable concisely; the "answer" should be 1–3 sentences.

QUESTION PHRASING PRINCIPLES:
- Vary your question style naturally based on the content and question type
- Ask WHAT is being done and WHY, without describing what is visible

EXAMPLES OF GOOD VS BAD QUESTIONS:

GOOD (varied, natural phrasing):
Q: "Based on the video clip, what probe/view acquisition approach is being used, and what key anatomical landmark is used to confirm the correct depth/orientation?"
Q: "What is the probe movement direction and maintained orientation, and what scanning objective does this achieve?"
Q: "The clip shows loss/obscuration of the target vessel. What optimization maneuver is performed, and what problem is it addressing to restore the view?"
Q: "At a branching point, what probe technique is used to improve clarity of the anatomy, and what imaging outcome does it produce?"
Q: "This clip shows a multi-step workflow segment. Summarize what the operator is trying to accomplish and the strategy used."
Q: "In this later workflow segment, what is the operator doing to ensure they are targeting the correct vessel, and how does the sequence of actions provide confirmation?"

BAD (gives away visual clues or contextual details in the question):
Q: "When the operator identifies the vertebral shadow behind the aorta in transverse view, what is the purpose of this landmark?"
Q: "When bowel gas obscures the view and the operator applies firm pressure, what is the goal?"
Q: "The operator sees two vessels and slides the probe to the right, isolating the IVC. Which vessel is which?"
Q: "In the sagittal plane, the operator notes that it can be confusing to tell the IVC from the aorta. Based on the clip sequence, what verification maneuver is used to resolve this ambiguity?"
Q: "When transitioning from transverse to longitudinal while keeping the target in view, what physical probe maneuver is performed?"
Q: "Based on the clip, what machine interaction is performed to start quantifying bladder size, and what is the goal of tapping on the image?"
Q: "The clip shows a visible change in how much anatomy fits on the display. What setting change is performed, and why does it help?"

TIME GROUNDING:
- Each item must include time_start and time_end aligned to the event(s) used:
  - Single event: use its exact start/end
  - Two or more adjacent events: time_start = min(starts), time_end = max(ends)

GROUNDTRUTH FIELD:
- Rephrase the action + interpretation into ONE integrated entry (1–3 sentences) that preserves all key details.
- Do not introduce details not supported by action/interpretation.

AVOID INVALID ITEMS:
- Do NOT reference audio, narration, or transcript.
- Do NOT ask for reading on-screen text or exact numeric readouts (e.g., "104 mm") unless it is guaranteed without OCR (assume it is not).
- If an action is a spoken instruction (e.g., "hold breath"), convert it into an EFFECT-based question that could be inferred visually (e.g., reduced respiratory motion / improved acoustic window), not "what instruction was said".

Return JSON only, matching the schema provided.
""".strip()


def build_user_prompt(events: List[Event], filename: str, n_questions: int) -> str:
    # Provide events as compact JSON for the model
    events_payload = [
        {
            "start": e.start,
            "end": e.end,
            "action": e.action,
            "interpretation": e.interpretation,
        }
        for e in events
    ]
    return f"""
You are given ground-truth events for: {filename}

TASK:
Generate {n_questions} Q/A items that strictly follow the system instructions and the 3 question types.

OUTPUT REQUIREMENTS:
- Return an object with a single key "items": an array of Q/A items.
- Each item must include: question, answer, question_type, groundtruth, time_start, time_end.
- Prefix question with [MCQ] or [FREE].

EVENTS (ground truth):
{json.dumps(events_payload, ensure_ascii=False)}
""".strip()


def qa_schema() -> Dict[str, Any]:
    # Structured Outputs JSON Schema
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                        "question_type": {"type": "string", "enum": QUESTION_TYPES},
                        "groundtruth": {"type": "string"},
                        "time_start": {"type": "number"},
                        "time_end": {"type": "number"},
                    },
                    "required": ["question", "answer", "question_type", "groundtruth", "time_start", "time_end"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def validate_and_filter_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for it in items:
        q = it["question"].strip()
        a = it["answer"].strip()
        gt = it["groundtruth"].strip()

        # Basic validity
        if it["question_type"] not in QUESTION_TYPES:
            continue
        if not (isinstance(it["time_start"], (int, float)) and isinstance(it["time_end"], (int, float))):
            continue
        if float(it["time_end"]) <= float(it["time_start"]):
            continue

        # Enforce muted/no-OCR constraints (best-effort)
        if BANNED_PHRASES_REGEX.search(q) or BANNED_PHRASES_REGEX.search(a):
            continue

        # Avoid numeric readout questions (best-effort)
        if NUMERIC_READOUT_REGEX.search(q):
            # if question asks about exact numeric value, drop it
            continue

        it["question"] = q
        it["answer"] = a
        it["groundtruth"] = gt
        cleaned.append(it)

    # Deduplicate by question text
    seen = set()
    deduped = []
    for it in cleaned:
        key = it["question"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    return deduped


def _extract_usage(resp: Any) -> Tuple[int, int, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return (0, 0, 0)
    if isinstance(usage, dict):
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
        return (input_tokens, output_tokens, total_tokens)

    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0)
    return (input_tokens, output_tokens, total_tokens)


def call_model_generate(
    client: OpenAI,
    model: str,
    events: List[Event],
    filename: str,
    n_questions: int,
    reasoning_effort: str,
    max_output_tokens: int,
    retries: int = 2,
) -> Tuple[List[Dict[str, Any]], Tuple[int, int, int]]:
    sys_prompt = build_system_prompt()
    user_prompt = build_user_prompt(events, filename, n_questions)

    for attempt in range(retries + 1):
        try:
            print(f"  Calling OpenAI API (attempt {attempt + 1}/{retries + 1})...")
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                reasoning={"effort": reasoning_effort},
                text={
                    "verbosity": "low",
                    "format": {
                        "type": "json_schema",
                        "name": "us_bench_qa",
                        "schema": qa_schema(),
                        "strict": True,
                    },
                },
                max_output_tokens=max_output_tokens,
            )

            data = json.loads(resp.output_text)
            items = data.get("items", [])
            if not isinstance(items, list):
                raise ValueError("Model output missing 'items' array.")
            print(f"  Received {len(items)} items from API")
            items = validate_and_filter_items(items)
            print(f"  {len(items)} items after validation/filtering")

            usage = _extract_usage(resp)

            # If too few survived filtering, ask again with fewer constraints? Instead: retry once.
            if len(items) < max(3, int(0.5 * n_questions)) and attempt < retries:
                user_prompt = user_prompt + "\n\nIMPORTANT: Many items were invalid before. Re-check muted/no-OCR rules and avoid numeric readouts."
                continue

            return items, usage

        except Exception as e:
            if attempt >= retries:
                raise
            time.sleep(1.5 * (attempt + 1))
            continue

    return [], (0, 0, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_dir",
        default="gt_all",
        help="Directory containing ground-truth *.json files.",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Directory to write QA files. Default: qa_generated",
    )
    ap.add_argument("--model", default="gpt-5.2-pro", help="OpenAI model id (default: gpt-5.2-pro).")
    ap.add_argument("--n_questions", type=int, default=24, help="Target number of QA items per input file.")
    ap.add_argument(
        "--reasoning_effort",
        default="high",
        choices=["none", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for GPT-5.2 family.",
    )
    ap.add_argument("--max_output_tokens", type=int, default=6000, help="Max output tokens per request.")
    ap.add_argument("--glob", default="*.json", help="Which files to read (glob pattern).")
    ap.add_argument("--parallel", type=int, default=3, help="Number of files to process in parallel (default: 3).")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser()
    if args.out_dir is None:
        out_dir = Path("qa_generated").resolve()
    else:
        out_dir = Path(args.out_dir).expanduser().resolve()

    if not in_dir.exists():
        print(f"Input dir not found: {in_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    paths = sorted(in_dir.glob(args.glob))
    if not paths:
        print(f"No files matched {args.glob} in {in_dir}", file=sys.stderr)
        sys.exit(1)

    # Filter out files that already have QA generated
    pending_paths = []
    for p in paths:
        out_name = p.name.replace("gt_", "qa_")
        out_path = out_dir / out_name
        if out_path.exists():
            print(f"⏭️  Skipping {p.name} (already exists: {out_name})")
        else:
            pending_paths.append(p)
    
    if not pending_paths:
        print(f"\n✅ All {len(paths)} files already have QA generated!")
        sys.exit(0)
    
    print(f"\n🚀 Processing {len(pending_paths)} files ({len(paths) - len(pending_paths)} skipped)...")
    print(f"   Using {args.parallel} parallel workers\n")

    def process_file(p: Path) -> Tuple[str, int, str, Tuple[int, int, int]]:
        """Process a single file and return (filename, item_count, status, usage)."""
        try:
            print(f"\n[START] {p.name}")
            print("=" * 60)
            events = load_events(p)
            if not events:
                return (p.name, 0, "skipped: no valid events", (0, 0, 0))

            # Calculate target questions based on event count: n_events + 1, capped at 20
            target_questions = min(len(events) + 1, 20)
            print(f"  [{p.name}] Target questions: {target_questions} (events: {len(events)}, formula: min(n+1, 20))")

            # Chunk if huge
            chunks = chunk_events(events)
            per_chunk = max(4, target_questions // max(1, len(chunks)))

            all_items: List[Dict[str, Any]] = []
            file_input_tokens = 0
            file_output_tokens = 0
            file_total_tokens = 0
            for ci, ev_chunk in enumerate(chunks):
                if len(chunks) > 1:
                    print(f"  [{p.name}] Processing chunk {ci + 1}/{len(chunks)} ({len(ev_chunk)} events)...")
                items, usage = call_model_generate(
                    client=client,
                    model=args.model,
                    events=ev_chunk,
                    filename=p.name if len(chunks) == 1 else f"{p.name} (chunk {ci+1}/{len(chunks)})",
                    n_questions=per_chunk,
                    reasoning_effort=args.reasoning_effort,
                    max_output_tokens=args.max_output_tokens,
                )
                all_items.extend(items)
                file_input_tokens += usage[0]
                file_output_tokens += usage[1]
                file_total_tokens += usage[2]

            # Final dedupe + trim to target
            all_items = validate_and_filter_items(all_items)
            if len(all_items) > target_questions:
                all_items = all_items[:target_questions]

            out_path = out_dir / p.name.replace("gt_", "qa_")
            out_path.write_text(json.dumps(all_items, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[DONE] {p.name} -> {len(all_items)} items")
            return (p.name, len(all_items), "success", (file_input_tokens, file_output_tokens, file_total_tokens))

        except Exception as e:
            print(f"[ERROR] {p.name}: {e}", file=sys.stderr)
            return (p.name, 0, f"error: {e}", (0, 0, 0))

    # Process files in parallel
    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        future_to_path = {executor.submit(process_file, p): p for p in pending_paths}
        for future in as_completed(future_to_path):
            result = future.result()
            total_input_tokens += result[3][0]
            total_output_tokens += result[3][1]
            total_tokens += result[3][2]
            results.append(result)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if r[2] == "success")
    total_items = sum(r[1] for r in results)
    print(f"✅ Successfully processed: {success_count}/{len(pending_paths)} files")
    print(f"📝 Total QA items generated: {total_items}")
    for name, count, status, _usage in sorted(results):
        emoji = "✅" if status == "success" else "⚠️" if "skipped" in status else "❌"
        print(f"   {emoji} {name}: {count} items ({status})")

    print("\nToken usage summary")
    print(f"  Input tokens:  {total_input_tokens}")
    print(f"  Output tokens: {total_output_tokens}")
    print(f"  Total tokens:  {total_tokens}")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
