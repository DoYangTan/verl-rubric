#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Row:
    input_text: str
    output_text: str
    score: float
    reward: float | None
    step: int | None


def _read_dotenv(repo_root: Path) -> dict[str, str]:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return {}
    out: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k:
            out[k] = v
    return out


def _resolve_jsonl_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(str(path))
    if not path.is_dir():
        raise FileNotFoundError(str(path))

    preferred = [path / "0.jsonl", path / "1.jsonl"]
    for p in preferred:
        if p.exists() and p.is_file():
            return p

    candidates = sorted([p for p in path.glob("*.jsonl") if p.is_file()])
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"no .jsonl files under {path}")


def _read_jsonl(path: Path, *, max_lines: int | None) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if max_lines is not None and line_no > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"{path}:{line_no}: invalid json: {e}") from e

            if not isinstance(obj, dict):
                raise RuntimeError(f"{path}:{line_no}: expected object, got {type(obj).__name__}")

            if "input" not in obj or "output" not in obj or "score" not in obj:
                missing = [k for k in ("input", "output", "score") if k not in obj]
                raise RuntimeError(f"{path}:{line_no}: missing keys: {missing}")

            input_text = str(obj["input"])
            output_text = str(obj["output"])
            score = float(obj["score"])
            reward = None
            if "reward" in obj and obj["reward"] is not None:
                try:
                    reward = float(obj["reward"])
                except Exception:
                    reward = None

            step = None
            if "step" in obj and obj["step"] is not None:
                try:
                    step = int(obj["step"])
                except Exception:
                    step = None

            rows.append(Row(input_text=input_text, output_text=output_text, score=score, reward=reward, step=step))
    return rows


def _quantile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return float("nan")
    if q <= 0:
        return sorted_xs[0]
    if q >= 1:
        return sorted_xs[-1]
    # Nearest-rank (inclusive) quantile.
    idx = int(round(q * (len(sorted_xs) - 1)))
    return sorted_xs[idx]


def _short(text: str, max_len: int) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _stats(label: str, rows: list[Row]) -> dict[str, Any]:
    scores = [r.score for r in rows]
    scores_sorted = sorted(scores)
    return {
        "label": label,
        "n": len(rows),
        "score_mean": statistics.mean(scores) if scores else float("nan"),
        "score_stdev": statistics.pstdev(scores) if len(scores) >= 2 else 0.0,
        "score_min": min(scores) if scores else float("nan"),
        "score_p50": _quantile(scores_sorted, 0.50),
        "score_p90": _quantile(scores_sorted, 0.90),
        "score_p99": _quantile(scores_sorted, 0.99),
        "score_max": max(scores) if scores else float("nan"),
        "step_values": sorted({r.step for r in rows if r.step is not None}),
        "reward_equals_score_ratio": (
            sum(1 for r in rows if r.reward is not None and abs(r.reward - r.score) < 1e-9) / len(rows)
            if rows
            else float("nan")
        ),
    }


def _index(rows: list[Row], key_fields: tuple[str, ...]) -> dict[tuple[Any, ...], Row]:
    out: dict[tuple[Any, ...], Row] = {}
    dup = 0
    for r in rows:
        key_parts: list[Any] = []
        for k in key_fields:
            if k == "input":
                key_parts.append(r.input_text)
            elif k == "output":
                key_parts.append(r.output_text)
            else:
                raise ValueError(f"unsupported key field: {k}")
        key = tuple(key_parts)
        if key in out:
            dup += 1
            continue
        out[key] = r
    return out


def _diff_report(
    a_label: str,
    a_rows: list[Row],
    b_label: str,
    b_rows: list[Row],
    match_on: tuple[str, ...],
    top_k: int,
) -> dict[str, Any]:
    a_map = _index(a_rows, match_on)
    b_map = _index(b_rows, match_on)
    keys_a = set(a_map)
    keys_b = set(b_map)
    common = keys_a & keys_b

    diffs = []
    for k in common:
        ra = a_map[k]
        rb = b_map[k]
        diffs.append(
            {
                "abs_diff": abs(ra.score - rb.score),
                "diff": ra.score - rb.score,
                "a_score": ra.score,
                "b_score": rb.score,
                "input": ra.input_text,
                "output": ra.output_text,
            }
        )

    diffs.sort(key=lambda d: d["abs_diff"], reverse=True)

    abs_diffs_sorted = sorted(d["abs_diff"] for d in diffs)
    return {
        "match_on": match_on,
        "a_only": len(keys_a - keys_b),
        "b_only": len(keys_b - keys_a),
        "common": len(common),
        "abs_diff_mean": statistics.mean(abs_diffs_sorted) if abs_diffs_sorted else float("nan"),
        "abs_diff_p50": _quantile(abs_diffs_sorted, 0.50),
        "abs_diff_p90": _quantile(abs_diffs_sorted, 0.90),
        "abs_diff_max": abs_diffs_sorted[-1] if abs_diffs_sorted else float("nan"),
        "top": diffs[:top_k],
    }


def _file_info(path: Path) -> str:
    try:
        st = path.stat()
        mtime = datetime.fromtimestamp(st.st_mtime).isoformat(sep=" ", timespec="seconds")
        return f"{path} (mtime {mtime}, {st.st_size} bytes)"
    except Exception:
        return str(path)


def _print_kv(title: str, d: dict[str, Any]) -> None:
    print(title)
    for k in sorted(d.keys()):
        print(f"  {k}: {d[k]}")


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Compare two validation jsonl dumps and diagnose whether differences come from "
            "generation outputs or reward/score computation."
        )
    )
    p.add_argument("--a", required=True, help="Path to jsonl A (e.g., GDPO step0 validation_log/*/0.jsonl)")
    p.add_argument("--b", required=True, help="Path to jsonl B (e.g., GRPO step0 validation_log/*/0.jsonl)")
    p.add_argument("--label-a", default="A", help="Label for A in the report")
    p.add_argument("--label-b", default="B", help="Label for B in the report")
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Show top-K largest score diffs (only prints short snippets)",
    )
    p.add_argument(
        "--snippet-len",
        type=int,
        default=220,
        help="Max chars for input/output snippet preview",
    )
    p.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="Read only the first N lines from each jsonl (0 means read all).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    a_path = _resolve_jsonl_path(Path(args.a))
    b_path = _resolve_jsonl_path(Path(args.b))

    print("== Files ==")
    print(f"- {args.label_a}: {_file_info(a_path)}")
    print(f"- {args.label_b}: {_file_info(b_path)}")

    repo_root = Path(__file__).resolve().parents[2]
    dotenv = _read_dotenv(repo_root)

    print("\n== Env (grader) ==")
    vllm_temp = os.getenv("VLLM_TEMPERATURE") or dotenv.get("VLLM_TEMPERATURE")
    vllm_temp_val = os.getenv("VLLM_TEMPERATURE_VAL") or dotenv.get("VLLM_TEMPERATURE_VAL")
    print(f"- VLLM_TEMPERATURE={vllm_temp!r}")
    print(f"- VLLM_TEMPERATURE_VAL={vllm_temp_val!r}")
    print(
        "  (If your scorer supports validate-only temperature, validation uses VLLM_TEMPERATURE_VAL; "
        "otherwise it falls back to VLLM_TEMPERATURE. Non-zero temperature makes the LLM-judge stochastic.)"
    )

    max_lines = None if int(args.max_lines) <= 0 else int(args.max_lines)
    a_rows = _read_jsonl(a_path, max_lines=max_lines)
    b_rows = _read_jsonl(b_path, max_lines=max_lines)

    print("\n== Basic stats ==")
    _print_kv(f"- {args.label_a}", _stats(args.label_a, a_rows))
    _print_kv(f"- {args.label_b}", _stats(args.label_b, b_rows))

    # 1) Compare by input (are we evaluating same prompts?)
    print("\n== Alignment by input ==")
    by_input = _diff_report(args.label_a, a_rows, args.label_b, b_rows, match_on=("input",), top_k=args.top_k)
    _print_kv("- Summary", {k: v for k, v in by_input.items() if k != "top"})
    if by_input["common"] == 0:
        print("No overlap by input. You are not comparing the same validation samples.")
        return 0

    # 2) Compare outputs for same input.
    a_by_input = _index(a_rows, ("input",))
    b_by_input = _index(b_rows, ("input",))
    common_inputs = set(a_by_input) & set(b_by_input)
    same_output = 0
    for k in common_inputs:
        if a_by_input[k].output_text == b_by_input[k].output_text:
            same_output += 1
    print(f"\n== Output equality on common inputs ==\n  same_output: {same_output}/{len(common_inputs)}")
    if same_output < len(common_inputs):
        print("  (If outputs differ, score differences are expected even with identical scoring.)")

    # 3) Compare score when BOTH input and output match (isolates scoring/judge differences).
    print("\n== Alignment by (input, output) (isolates scoring differences) ==")
    by_io = _diff_report(args.label_a, a_rows, args.label_b, b_rows, match_on=("input", "output"), top_k=args.top_k)
    _print_kv("- Summary", {k: v for k, v in by_io.items() if k != "top"})

    if by_io["common"] == 0:
        print("No overlap by (input, output). The two runs generated completely different outputs.")
        return 0

    print("\n== Top diffs (by input, output) ==")
    for d in by_io["top"]:
        print(
            f"- abs_diff={d['abs_diff']:.6f} diff={d['diff']:.6f} "
            f"{args.label_a}={d['a_score']:.6f} {args.label_b}={d['b_score']:.6f}"
        )
        print(f"  input:  {_short(d['input'], args.snippet_len)}")
        print(f"  output: {_short(d['output'], args.snippet_len)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
