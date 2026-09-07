import argparse
import csv
import json

import pytest

import cli
from tests.conftest import FakeClient, KINEMATICS_JSON


# ---------------------------------------------------------------- _load_problems

def test_load_problems_from_plain_string_list(tmp_path):
    p = tmp_path / "problems.json"
    p.write_text(json.dumps(["problem one", "problem two"]))
    problems = cli._load_problems(str(p), None)
    assert problems == [{"text": "problem one", "known_context": None},
                          {"text": "problem two", "known_context": None}]


def test_load_problems_from_object_list_with_known_context(tmp_path):
    p = tmp_path / "problems.json"
    p.write_text(json.dumps([{"text": "problem one", "known_context": "d = 5"}]))
    problems = cli._load_problems(str(p), None)
    assert problems == [{"text": "problem one", "known_context": "d = 5"}]


def test_load_problems_from_text_arg_ignores_file():
    problems = cli._load_problems(None, "a direct problem")
    assert problems == [{"text": "a direct problem", "known_context": None}]


def test_load_problems_raises_on_missing_input():
    with pytest.raises(ValueError):
        cli._load_problems(None, None)


def test_load_problems_raises_on_non_array_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"not": "an array"}))
    with pytest.raises(ValueError):
        cli._load_problems(str(p), None)


def test_load_problems_raises_on_unrecognized_entry(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([42]))
    with pytest.raises(ValueError):
        cli._load_problems(str(p), None)


# ---------------------------------------------------------------- _output_format / _write_output

def test_output_format_inferred_from_json_extension():
    assert cli._output_format("out.json", "csv") == "json"


def test_output_format_inferred_from_csv_extension():
    assert cli._output_format("out.csv", "json") == "csv"


def test_output_format_falls_back_to_explicit_when_no_recognizable_extension():
    assert cli._output_format("out.dat", "json") == "json"


def test_write_output_csv_to_file(tmp_path):
    out = tmp_path / "results.csv"
    cli._write_output([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], str(out), "csv")
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]


def test_write_output_json_to_file(tmp_path):
    out = tmp_path / "results.json"
    cli._write_output([{"a": 1}], str(out), "json")
    assert json.loads(out.read_text()) == [{"a": 1}]


def test_write_output_csv_to_stdout(capsys):
    cli._write_output([{"a": 1, "b": "x"}], None, "csv")
    captured = capsys.readouterr()
    assert "a,b" in captured.out
    assert "1,x" in captured.out


def test_write_output_empty_rows_does_not_crash(tmp_path):
    out = tmp_path / "empty.csv"
    cli._write_output([], str(out), "csv")
    assert not out.exists()  # nothing to write, and nothing should blow up either


# ---------------------------------------------------------------- cmd_solve

def _args(**kwargs):
    defaults = {"problems": None, "text": None, "output": None, "format": "csv"}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_cmd_solve_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "LMStudioClient",
                          lambda: FakeClient(payload_json=KINEMATICS_JSON, final_answers={"a": 2.0}))
    out = tmp_path / "results.csv"
    args = _args(text="A car accelerates from 8 m/s to 20 m/s in 6 seconds.", output=str(out))
    rc = cli.cmd_solve(args)
    assert rc == 0
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["domain"] == "kinematics"
    assert rows[0]["target"] == "a"
    assert float(rows[0]["value"]) == pytest.approx(2.0)
    assert rows[0]["error"] == ""


def test_cmd_solve_reports_connection_failure(monkeypatch):
    class DownClient:
        def is_available(self):
            return False, "LM Studio is not reachable."
    monkeypatch.setattr(cli, "LMStudioClient", DownClient)
    args = _args(text="a problem")
    rc = cli.cmd_solve(args)
    assert rc == 1


def test_cmd_solve_captures_per_problem_errors_without_aborting_the_batch(monkeypatch, tmp_path):
    def _raise(*a, **kw):
        raise RuntimeError("extraction blew up")

    monkeypatch.setattr(cli, "LMStudioClient",
                          lambda: FakeClient(payload_json=KINEMATICS_JSON, final_answers={"a": 2.0}))
    monkeypatch.setattr(cli, "extract_model", _raise)
    out = tmp_path / "results.json"
    args = _args(text="a problem", output=str(out))
    rc = cli.cmd_solve(args)
    assert rc == 0  # a per-problem error shouldn't fail the whole batch
    rows = json.loads(out.read_text())
    assert "extraction blew up" in rows[0]["error"]


def test_cmd_solve_multiple_problems_from_file(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "LMStudioClient",
                          lambda: FakeClient(payload_json=KINEMATICS_JSON, final_answers={"a": 2.0}))
    problems_file = tmp_path / "problems.json"
    problems_file.write_text(json.dumps(["problem A", "problem B", "problem C"]))
    out = tmp_path / "results.csv"
    args = _args(problems=str(problems_file), output=str(out))
    rc = cli.cmd_solve(args)
    assert rc == 0
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3


# ---------------------------------------------------------------- cmd_montecarlo

def _mc_args(**kwargs):
    defaults = {"problem": None, "text": None, "target": "a", "uncertain": ["v:1.0"],
                "samples": 200, "seed": 42, "output": None, "format": "csv"}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_cmd_montecarlo_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "LMStudioClient",
                          lambda: FakeClient(payload_json=KINEMATICS_JSON, final_answers={"a": 2.0}))
    out = tmp_path / "samples.csv"
    args = _mc_args(text="A car accelerates from 8 m/s to 20 m/s in 6 seconds.", output=str(out))
    rc = cli.cmd_montecarlo(args)
    assert rc == 0
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    # (v - u)/t with v=20±1, u=8, t=6 -- mean should land near 2.0
    values = [float(r["sample"]) for r in rows]
    assert sum(values) / len(values) == pytest.approx(2.0, abs=0.3)


def test_cmd_montecarlo_reports_unknown_symbol(monkeypatch, capsys):
    monkeypatch.setattr(cli, "LMStudioClient",
                          lambda: FakeClient(payload_json=KINEMATICS_JSON, final_answers={"a": 2.0}))
    args = _mc_args(text="a problem", uncertain=["not_a_real_symbol:1.0"])
    rc = cli.cmd_montecarlo(args)
    assert rc == 1
    assert "not_a_real_symbol" in capsys.readouterr().err


def test_cmd_montecarlo_reports_malformed_uncertain_spec(monkeypatch, capsys):
    monkeypatch.setattr(cli, "LMStudioClient",
                          lambda: FakeClient(payload_json=KINEMATICS_JSON, final_answers={"a": 2.0}))
    args = _mc_args(text="a problem", uncertain=["v_no_colon"])
    rc = cli.cmd_montecarlo(args)
    assert rc == 1
    assert "symbol:std" in capsys.readouterr().err


def test_cmd_montecarlo_seed_is_reproducible(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "LMStudioClient",
                          lambda: FakeClient(payload_json=KINEMATICS_JSON, final_answers={"a": 2.0}))
    out1 = tmp_path / "run1.csv"
    out2 = tmp_path / "run2.csv"
    cli.cmd_montecarlo(_mc_args(text="a problem", output=str(out1), seed=123))
    cli.cmd_montecarlo(_mc_args(text="a problem", output=str(out2), seed=123))
    assert out1.read_text() == out2.read_text()


def test_cmd_montecarlo_connection_failure(monkeypatch):
    class DownClient:
        def is_available(self):
            return False, "down"
    monkeypatch.setattr(cli, "LMStudioClient", DownClient)
    rc = cli.cmd_montecarlo(_mc_args(text="a problem"))
    assert rc == 1


# ---------------------------------------------------------------- _resolve_montecarlo_problem_text

def test_resolve_montecarlo_problem_text_prefers_text_arg():
    assert cli._resolve_montecarlo_problem_text("ignored.txt", "direct text") == "direct text"


def test_resolve_montecarlo_problem_text_reads_plain_text_file(tmp_path):
    p = tmp_path / "problem.txt"
    p.write_text("a plain text problem")
    assert cli._resolve_montecarlo_problem_text(str(p), None) == "a plain text problem"


def test_resolve_montecarlo_problem_text_reads_json_array_first_entry(tmp_path):
    p = tmp_path / "problems.json"
    p.write_text(json.dumps(["first problem", "second problem"]))
    assert cli._resolve_montecarlo_problem_text(str(p), None) == "first problem"


def test_resolve_montecarlo_problem_text_reads_json_object_array(tmp_path):
    p = tmp_path / "problems.json"
    p.write_text(json.dumps([{"text": "first problem"}]))
    assert cli._resolve_montecarlo_problem_text(str(p), None) == "first problem"


# ---------------------------------------------------------------- argument parsing

def test_build_parser_requires_a_subcommand():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_parser_solve_accepts_text_flag():
    parser = cli.build_parser()
    args = parser.parse_args(["solve", "--text", "a problem"])
    assert args.text == "a problem"
    assert args.func is cli.cmd_solve


def test_build_parser_montecarlo_requires_target_and_uncertain():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["montecarlo", "problem.txt"])  # missing --target/--uncertain


def test_build_parser_montecarlo_accepts_repeated_uncertain():
    parser = cli.build_parser()
    args = parser.parse_args(["montecarlo", "problem.txt", "--target", "a",
                                "--uncertain", "v:1.0", "--uncertain", "t:0.2"])
    assert args.uncertain == ["v:1.0", "t:0.2"]
