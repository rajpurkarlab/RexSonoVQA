#!/usr/bin/env python3
"""
Refine MCQ distractors for QA files.

- Reads QA JSON files from qa_generated6
- For MCQ items, rewrites ONLY the distractor options (keeps stem + correct option text + answer letter)
- FREE questions are left untouched
- Writes results to qa_generated6_mcq
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI


MCQ_OPTION_RE = re.compile(r"^([A-D])[\.|\)]\s+(.*)$")


def parse_mcq(question: str) -> Tuple[str, Dict[str, str], str]:
	"""
	Returns (stem, options, separator) or raises ValueError if parsing fails.
	separator is '.' or ')' depending on original options.
	"""
	lines = question.strip().splitlines()
	stem_lines: List[str] = []
	options: Dict[str, str] = {}
	separator = "."

	for line in lines:
		m = MCQ_OPTION_RE.match(line.strip())
		if m:
			letter, text = m.group(1), m.group(2)
			options[letter] = text.strip()
			if ")" in line:
				separator = ")"
			else:
				separator = "."
		else:
			if not options:
				stem_lines.append(line)

	if len(options) != 4:
		raise ValueError("MCQ options not found or not exactly 4.")

	stem = "\n".join(stem_lines).strip()
	return stem, options, separator


def build_question(stem: str, options: Dict[str, str], separator: str) -> str:
	lines = [stem]
	for letter in ["A", "B", "C", "D"]:
		lines.append(f"{letter}{separator} {options[letter]}")
	return "\n".join(lines)


def sample_distractors(pool: List[str], k: int, exclude: str) -> List[str]:
	candidates = [p for p in pool if p and p.strip() and p.strip() != exclude.strip()]
	if len(candidates) <= k:
		return candidates
	return random.sample(candidates, k)


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


def refine_distractors(
	client: OpenAI,
	model: str,
	stem: str,
	correct_text: str,
	distractor_letters: List[str],
	exemplar_options: List[str],
) -> Tuple[Dict[str, str], Tuple[int, int, int]]:
	"""
	Ask the model to generate distractors for the given letters.
	Returns mapping letter -> option text.
	"""
	system_prompt = """You are a medical MCQ editor. Improve distractors so they are clinically plausible and not trivially wrong.

Rules:
- Do NOT change the question stem.
- Do NOT change the correct option text.
- Provide only the requested distractors.
- Distractors must be medically plausible and consistent with the exam context, but clearly incorrect.
- Avoid nonsensical or unrelated anatomy/procedures.
- Keep each option concise (1 sentence max).
- Do not include letters like 'A.' in the option text; return raw option text only.
""".strip()

	user_payload = {
		"stem": stem,
		"correct_option": correct_text,
		"distractor_letters": distractor_letters,
		"exemplar_options": exemplar_options,
	}

	resp = client.responses.create(
		model=model,
		input=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
		],
		text={
			"format": {
				"type": "json_schema",
				"name": "mcq_distractors",
				"schema": {
					"type": "object",
					"properties": {
						"options": {
							"type": "object",
							"properties": {letter: {"type": "string"} for letter in distractor_letters},
							"required": distractor_letters,
							"additionalProperties": False,
						}
					},
					"required": ["options"],
					"additionalProperties": False,
				},
				"strict": True,
			}
		},
		max_output_tokens=800,
	)

	data = json.loads(resp.output_text)
	usage = _extract_usage(resp)
	return data.get("options", {}), usage


def process_file(
	p: Path,
	out_dir: Path,
	option_pool: List[str],
	model: str,
) -> Tuple[str, int, Tuple[int, int, int]]:
	client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
	items = json.loads(p.read_text(encoding="utf-8"))
	changed = 0
	file_input_tokens = 0
	file_output_tokens = 0
	file_total_tokens = 0
	processed_mcq = 0
	print(f"[START] {p.name}: {len(items)} items")

	for item in items:
		q = item.get("question", "")
		if not (isinstance(q, str) and q.startswith("[MCQ]")):
			continue

		answer_letter = str(item.get("answer", "")).strip().upper()[:1]
		if answer_letter not in {"A", "B", "C", "D"}:
			continue

		try:
			stem, options, separator = parse_mcq(q)
		except Exception:
			continue

		correct_text = options[answer_letter]
		distractor_letters = [l for l in ["A", "B", "C", "D"] if l != answer_letter]
		exemplars = sample_distractors(option_pool, k=12, exclude=correct_text)

		try:
			new_distractors, usage = refine_distractors(
				client=client,
				model=model,
				stem=stem,
				correct_text=correct_text,
				distractor_letters=distractor_letters,
				exemplar_options=exemplars,
			)
		except Exception as e:
			print(f"[WARN] {p.name}: failed to refine MCQ: {e}")
			continue

		file_input_tokens += usage[0]
		file_output_tokens += usage[1]
		file_total_tokens += usage[2]
		processed_mcq += 1
		if processed_mcq % 10 == 0:
			print(f"  {p.name}: processed {processed_mcq} MCQs...")

		# Update options while keeping correct option text intact
		updated = False
		for letter in distractor_letters:
			new_text = new_distractors.get(letter, "").strip()
			if new_text and new_text != options[letter]:
				options[letter] = new_text
				updated = True

		if updated:
			item["question"] = build_question(stem, options, separator)
			changed += 1

	out_path = out_dir / p.name
	out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
	return (p.name, changed, (file_input_tokens, file_output_tokens, file_total_tokens))


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument(
		"--in_dir",
		default="qa_generated",
		help="Input QA directory",
	)
	ap.add_argument(
		"--out_dir",
		default="qa_refined",
		help="Output directory",
	)
	ap.add_argument("--model", default="gpt-5.2-pro", help="OpenAI model")
	ap.add_argument("--glob", default="*.json", help="Which files to read")
	ap.add_argument("--seed", type=int, default=13, help="Random seed for sampling")
	ap.add_argument("--parallel", type=int, default=6, help="Number of files to process concurrently")
	args = ap.parse_args()

	random.seed(args.seed)

	in_dir = Path(args.in_dir).expanduser().resolve()
	out_dir = Path(args.out_dir).expanduser().resolve()
	if not in_dir.exists():
		print(f"Input dir not found: {in_dir}", file=sys.stderr)
		sys.exit(1)
	out_dir.mkdir(parents=True, exist_ok=True)

	qa_paths = sorted(in_dir.glob(args.glob))
	if not qa_paths:
		print(f"No files matched {args.glob} in {in_dir}", file=sys.stderr)
		sys.exit(1)

	# Skip files that already exist in output directory
	pending_paths: List[Path] = []
	for p in qa_paths:
		out_path = out_dir / p.name
		if out_path.exists():
			print(f"⏭️  Skipping {p.name} (already exists)")
		else:
			pending_paths.append(p)

	if not pending_paths:
		print(f"\n✅ All {len(qa_paths)} files already exist in {out_dir}!")
		sys.exit(0)

	# Build a global pool of MCQ option texts for distractor inspiration
	option_pool: List[str] = []
	for p in qa_paths:
		try:
			items = json.loads(p.read_text(encoding="utf-8"))
		except Exception:
			continue
		for item in items:
			q = item.get("question", "")
			if isinstance(q, str) and q.startswith("[MCQ]"):
				try:
					_, opts, _ = parse_mcq(q)
					option_pool.extend(opts.values())
				except Exception:
					continue

	grand_input = 0
	grand_output = 0
	grand_total = 0
	with ThreadPoolExecutor(max_workers=args.parallel) as executor:
		future_to_path = {
			executor.submit(process_file, p, out_dir, option_pool, args.model): p for p in pending_paths
		}
		for future in as_completed(future_to_path):
			name, changed, usage = future.result()
			grand_input += usage[0]
			grand_output += usage[1]
			grand_total += usage[2]
			print(f"Wrote {name} (MCQ updated: {changed})")

	print("\nToken usage summary")
	print(f"  Input tokens:  {grand_input}")
	print(f"  Output tokens: {grand_output}")
	print(f"  Total tokens:  {grand_total}")


if __name__ == "__main__":
	main()
