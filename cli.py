#!/usr/bin/env python3
"""
Command-line entry point for the Math Representation System, reusing
the exact same modules/ the Streamlit app itself is built on --
equation_engine, verifier, monte_carlo -- with no Streamlit dependency
at all. app.py is really just one consumer of that library; this is
another. Built for the workflows point-and-click can't reasonably
serve: batch-processing dozens or hundreds of problem variants
overnight, wiring this into a researcher's own Python/pandas pipeline,
or running it as a scripted regression check against a formula library
in CI.

Usage:
    python cli.py solve problems.json --output results.csv
    python cli.py solve --text "A car accelerates from 8 m/s to 20 m/s in 6 seconds. Find a." -o result.json
    python cli.py montecarlo problem.txt --target a --uncertain v_f:1.0 --samples 5000 --seed 42 -o samples.csv

`problems.json` (for `solve`) is either a plain JSON array of problem-
text strings, or an array of {"text": "...", "known_context": "..."}
objects for more control per problem (known_context mirrors what the
Variable Workspace passes along in the UI -- previously-solved values
the extractor should treat as known inputs). Requires LM Studio running
with a model loaded, same as the app itself -- this is a different
front end onto the same pipeline, not an offline mode.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from modules.llm_client import LMStudioClient
from modules.equation_engine import extract_model, target_kind
from modules.verifier import verify
from modules.monte_carlo import run_monte_carlo, UncertainVariable


def _load_problems(problems_path: str | None, text: str | None) -> list[dict]:
    """Returns a list of {"text": ..., "known_context": ...} dicts
    regardless of which input form was given."""
    if text:
        return [{"text": text, "known_context": None}]
    if not problems_path:
        raise ValueError("Give either a problems file or --text.")
    raw = json.loads(Path(problems_path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{problems_path} must contain a JSON array.")
    problems = []
    for item in raw:
        if isinstance(item, str):
            problems.append({"text": item, "known_context": None})
        elif isinstance(item, dict) and "text" in item:
            problems.append({"text": item["text"], "known_context": item.get("known_context")})
        else:
            raise ValueError(f"Unrecognized problem entry: {item!r} -- expected a string or "
                              "an object with a 'text' key.")
    return problems


def _output_format(output_path: str | None, explicit_format: str) -> str:
    if output_path and output_path.lower().endswith(".json"):
        return "json"
    if output_path and output_path.lower().endswith(".csv"):
        return "csv"
    return explicit_format


def _write_output(rows: list[dict], output_path: str | None, fmt: str):
    """Writes `rows` (a list of flat dicts) as CSV or JSON, to a file if
    `output_path` is given, else to stdout -- so this composes cleanly
    with shell pipelines (`python cli.py solve x.json | jq .`) as well
    as writing a file directly."""
    resolved_format = _output_format(output_path, fmt)
    if resolved_format == "json":
        text = json.dumps(rows, indent=2, default=str)
        if output_path:
            Path(output_path).write_text(text)
        else:
            print(text)
        return

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    if output_path:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cmd_solve(args) -> int:
    client = LMStudioClient()
    ok, msg = client.is_available()
    if not ok:
        print(f"Error: {msg}", file=sys.stderr)
        return 1

    try:
        problems = _load_problems(args.problems, args.text)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    rows = []
    for i, p in enumerate(problems, start=1):
        print(f"[{i}/{len(problems)}] Solving: {p['text'][:70]}", file=sys.stderr)
        row = {"index": i, "problem_text": p["text"], "domain": None, "target": None,
               "value": None, "confidence": None, "passed": None, "error": None}
        try:
            model = extract_model(client, p["text"], known_context=p.get("known_context"))
            report = verify(model, client, p["text"])
            row["domain"] = model.problem_domain
            row["passed"] = report.passed
            conf_label, _ = report.confidence()
            row["confidence"] = conf_label
            algebraic_targets = [t for t in model.solve_for if target_kind(model, t) == "equation"]
            if algebraic_targets:
                target = algebraic_targets[0]
                row["target"] = target
                row["value"] = report.sympy_numeric_answers.get(target)
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)

    _write_output(rows, args.output, args.format)
    n_ok = sum(1 for r in rows if r["error"] is None and r["passed"])
    print(f"Done: {n_ok}/{len(rows)} solved and verified.", file=sys.stderr)
    return 0


def _resolve_montecarlo_problem_text(problem_arg: str, text_arg: str | None) -> str:
    if text_arg:
        return text_arg
    raw = Path(problem_arg).read_text()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw  # a plain text file
    if isinstance(data, list) and data:
        first = data[0]
        return first["text"] if isinstance(first, dict) else first
    return raw


def cmd_montecarlo(args) -> int:
    client = LMStudioClient()
    ok, msg = client.is_available()
    if not ok:
        print(f"Error: {msg}", file=sys.stderr)
        return 1

    problem_text = _resolve_montecarlo_problem_text(args.problem, args.text)
    model = extract_model(client, problem_text)
    verify(model, client, problem_text)  # populate the model fully; the report itself isn't needed here

    uncertain_vars = []
    for spec in args.uncertain:
        if ":" not in spec:
            print(f"Error: '--uncertain {spec}' must be of the form symbol:std", file=sys.stderr)
            return 1
        symbol, std_text = spec.split(":", 1)
        var = next((v for v in model.variables if v.symbol == symbol), None)
        if var is None or var.known_value is None:
            print(f"Error: '{symbol}' isn't a known input of this problem.", file=sys.stderr)
            return 1
        uncertain_vars.append(UncertainVariable(symbol=symbol, mean=var.known_value, std=float(std_text)))

    try:
        result = run_monte_carlo(model, args.target, uncertain_vars, n_samples=args.samples, seed=args.seed)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"seed={result.seed} mean={result.mean:.6g} std={result.std:.4g} "
          f"p5={result.p5:.6g} p95={result.p95:.6g} n_failed={result.n_failed} "
          f"(reuse --seed {result.seed} to reproduce this exact run)", file=sys.stderr)

    if args.output:
        _write_output([{"sample": v} for v in result.samples], args.output, args.format)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mathrep",
        description="Command-line access to the Math Representation System's solving/"
                     "verification pipeline -- for batch runs and scripting, without a browser.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve_parser = subparsers.add_parser("solve", help="Extract, verify, and solve one or more problems.")
    solve_parser.add_argument("problems", nargs="?", help="Path to a JSON file of problem texts.")
    solve_parser.add_argument("--text", help="Solve a single problem given directly on the command line.")
    solve_parser.add_argument("--output", "-o", help="Output file path (CSV or JSON, inferred from "
                                                        "the extension, or --format if it can't be).")
    solve_parser.add_argument("--format", choices=["csv", "json"], default="csv")
    solve_parser.set_defaults(func=cmd_solve)

    mc_parser = subparsers.add_parser(
        "montecarlo", help="Run Monte Carlo uncertainty propagation on a single problem.")
    mc_parser.add_argument("problem", help="Path to a text file (or a single/multi-problem JSON "
                                             "file, first entry used) with the problem.")
    mc_parser.add_argument("--text", help="Give the problem text directly instead of a file.")
    mc_parser.add_argument("--target", required=True, help="Which solve_for target to propagate to.")
    mc_parser.add_argument("--uncertain", action="append", required=True, metavar="symbol:std",
                             help="Repeatable -- e.g. --uncertain v_f:1.0 --uncertain t:0.2")
    mc_parser.add_argument("--samples", type=int, default=1000)
    mc_parser.add_argument("--seed", type=int, default=None,
                             help="Omit for a fresh random seed (printed to stderr for later reuse).")
    mc_parser.add_argument("--output", "-o", help="Write the raw samples to this file (CSV or JSON).")
    mc_parser.add_argument("--format", choices=["csv", "json"], default="csv")
    mc_parser.set_defaults(func=cmd_montecarlo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
