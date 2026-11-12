"""Unit and integration tests for the P3 inference harness (`src/harness.py` and `scripts/run_units.py`).

All file-writing tests use pytest `tmp_path` and never touch the repository `results/` directory.
"""

from __future__ import annotations

from collections.abc import Sequence
import csv
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_invariants import check_invariants, sha256_bytes  # noqa: E402
from scripts.run_units import build_arg_parser, main as run_units_main  # noqa: E402
from src.harness import (  # noqa: E402
    Backend,
    RunUnit,
    StubBackend,
    assemble_rag_passage_ids,
    build_messages,
    compute_study_hashes,
    extract_answer,
    load_extraction_config,
    load_prompt_config,
    load_result_rows,
    parse_run_unit,
    parse_units_list,
    render_closed_book_text,
    render_rag_text,
    render_unit_messages,
    resolve_results_dir,
    run_unit,
    sha256_text,
)


class FakeOfficialBackend(Backend):
    """Non-stub mock backend used in `tmp_path` sandbox tests to verify official-run ledger behavior."""

    def __init__(self, model_name: str = "Qwen3-8B") -> None:
        self.model_name = model_name
        self.backend_name = "mlx"
        self.is_stub = False
        self.model_id = f"official/{model_name}"

    def format_prompt(self, messages: Sequence[Mapping[str, str]]) -> str:
        content = str(messages[0]["content"])
        return f"<|official|{self.model_name}|>\n{content}\n<|assistant|>\n"

    def generate(
        self,
        prompt_texts: Sequence[str],
        max_new_tokens: int,
        greedy: bool = True,
    ) -> list[str]:
        return [f"Answer: official_{sha256_text(p)[:8]}" for p in prompt_texts]


def _make_synthetic_input_row(
    qid: str,
    split: str = "pilot",
    seed: int = 0,
    gold_pid: list[int] | None = None,
    gold_positions: dict[str, int] | None = None,
) -> dict[str, Any]:
    g_pid = gold_pid if gold_pid is not None else [0, 0]
    g_pos = gold_positions if gold_positions is not None else {"1": 0, "2": 1, "4": 2, "8": 4}
    hard_pids = [[1, i] for i in range(8)]
    rand_pids = [[2, i] for i in range(8)]
    return {
        "question_id": qid,
        "split": split,
        "seed": seed,
        "article_index": g_pid[0],
        "article_title": "Test_Article",
        "question_text": f"What is the answer for {qid}?",
        "gold_answers": [f"gold_{qid}"],
        "gold_paragraph_id": g_pid,
        "gold_score": 0.75,
        "hard_pool_paragraph_ids": [[1, i] for i in range(20)],
        "hard_paragraph_ids": hard_pids,
        "hard_candidate_scores": [0.5 - 0.01 * i for i in range(8)],
        "random_paragraph_ids": rand_pids,
        "random_length_match_windows": [5] * 8,
        "p": 0.5,
        "gold_positions": g_pos,
        "token_lengths": {
            "qwen3": {"gold": 50, "hard": [50] * 8, "random": [50] * 8},
            "llama3": {"gold": 48, "hard": [48] * 8, "random": [48] * 8},
        },
        "rejection_counts": {
            "candidate_depth": 100,
            "hard_top_k_leak_rejected": 1,
            "hard_before_pool_k_leak_rejected": 1,
            "hard_top_k_non_rejected": 99,
            "random_pool_non_gold_article_total": 1100,
            "random_pool_leak_rejected": 2,
        },
    }


@pytest.fixture()
def sandbox_repo(tmp_path: Path) -> Path:
    """Create an isolated temporary repository with configs, raw paragraphs, and input seed files."""
    for subdir in (
        "configs",
        "tests",
        "data/raw",
        "data/inputs",
        "data/sampling",
        " data/frozen",
    ):
        (tmp_path / subdir.strip()).mkdir(parents=True, exist_ok=True)

    for rel in (
        "configs/prereg.yaml",
        "configs/prompt_v1.yaml",
        "configs/extraction_v1.yaml",
        "configs/runtime.yaml",
        "tests/prereg.sha256",
        "tests/prompt.sha256",
        "requirements.txt",
        "requirements-mac.txt",
    ):
        src_file = REPO_ROOT / rel
        if src_file.exists():
            shutil.copyfile(src_file, tmp_path / rel)

    # Build a synthetic dev-v2.0.json with 3 articles x 20 paragraphs
    articles = []
    for art_idx, title in enumerate(["Gold_Art", "Hard_Art", "Random_Art"]):
        paragraphs = []
        for para_idx in range(20):
            paragraphs.append(
                {
                    "context": f"Passage text for [{art_idx}, {para_idx}].",
                    "qas": [],
                }
            )
        articles.append({"title": title, "paragraphs": paragraphs})
    (tmp_path / "data" / "raw" / "dev-v2.0.json").write_text(
        json.dumps({"version": "v2.0", "data": articles}),
        encoding="utf-8",
    )

    # Create synthetic sampling manifest and split files
    (tmp_path / "data" / "sampling" / "manifest.json").write_text(
        json.dumps({"seed": 4207, "split_counts": {"pilot": 4, "main": 4, "reserve": 2}}),
        encoding="utf-8",
    )

    for split in ("pilot", "main"):
        all_qids = [f"{split}_q{i}" for i in range(1, 5)]
        (tmp_path / "data" / "sampling" / f"{split}.json").write_text(
            json.dumps([{"question_id": qid} for qid in all_qids]),
            encoding="utf-8",
        )
        # Seed 0 has q1, q2, q3 (q4 excluded in seed 0)
        # Seed 1 has q1, q2, q3, q4 (all 4 present)
        # Seed 2 has q1, q3, q4 (q2 excluded in seed 2)
        seed_qids = {
            0: [f"{split}_q1", f"{split}_q2", f"{split}_q3"],
            1: [f"{split}_q1", f"{split}_q2", f"{split}_q3", f"{split}_q4"],
            2: [f"{split}_q1", f"{split}_q3", f"{split}_q4"],
        }
        for seed, qids in seed_qids.items():
            rows = [_make_synthetic_input_row(qid, split=split, seed=seed) for qid in qids]
            seed_file = tmp_path / "data" / "inputs" / f"{split}_seed{seed}.jsonl"
            seed_file.write_text(
                "".join(json.dumps(r) + "\n" for r in rows),
                encoding="utf-8",
            )

    return tmp_path


# ---------------------------------------------------------------------------
# 1. Prompt Renderer Tests
# ---------------------------------------------------------------------------


def test_rag_prompt_matches_3passages_fixture_byte_for_byte() -> None:
    """RAG renderer output for AAA/BBB/CCC/QQQ must equal tests/fixtures/rag_prompt_3passages.txt byte for byte."""
    fixture_path = REPO_ROOT / "tests" / "fixtures" / "rag_prompt_3passages.txt"
    prompt_cfg = load_prompt_config(REPO_ROOT / "configs" / "prompt_v1.yaml")

    rendered = render_rag_text(["AAA", "BBB", "CCC"], "QQQ", prompt_cfg=prompt_cfg)
    assert rendered.encode("utf-8") == fixture_path.read_bytes()


def test_renderer_snapshots_closed_book_n0_and_n2() -> None:
    """Snapshot test against configs/prompt_v1.yaml for closed-book, n=0 (oracle), and n=2 (RAG)."""
    prompt_cfg = load_prompt_config(REPO_ROOT / "configs" / "prompt_v1.yaml")

    # Closed-book snapshot
    cb_text = render_closed_book_text("Who discovered penicillin?", prompt_cfg=prompt_cfg)
    assert cb_text == (
        "Question: Who discovered penicillin?\n"
        "Answer with the shortest possible answer, no explanation."
    )
    cb_msgs = build_messages(cb_text)
    assert cb_msgs == [{"role": "user", "content": cb_text}]

    # n=0 (oracle, 1 passage) snapshot
    n0_text = render_rag_text(
        ["Alexander Fleming discovered penicillin in 1928."],
        "Who discovered penicillin?",
        prompt_cfg=prompt_cfg,
    )
    assert n0_text == (
        "Context:\n"
        "[Document 1] Alexander Fleming discovered penicillin in 1928.\n"
        "\n"
        "Question: Who discovered penicillin?\n"
        "Answer with the shortest text span that answers the question, no explanation."
    )
    n0_msgs = build_messages(n0_text)
    assert n0_msgs == [{"role": "user", "content": n0_text}]

    # n=2 (1 gold + 2 distractors = 3 passages) snapshot
    n2_text = render_rag_text(
        [
            "Distractor passage one.",
            "Alexander Fleming discovered penicillin in 1928.",
            "Distractor passage two.",
        ],
        "Who discovered penicillin?",
        prompt_cfg=prompt_cfg,
    )
    assert n2_text == (
        "Context:\n"
        "[Document 1] Distractor passage one.\n"
        "[Document 2] Alexander Fleming discovered penicillin in 1928.\n"
        "[Document 3] Distractor passage two.\n"
        "\n"
        "Question: Who discovered penicillin?\n"
        "Answer with the shortest text span that answers the question, no explanation."
    )
    n2_msgs = build_messages(n2_text)
    assert n2_msgs == [{"role": "user", "content": n2_text}]


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_gold_position_insertion_all_slots(n: int) -> None:
    """Verify gold_paragraph_id is inserted at exact 0-based index gold_positions[str(n)] among n+1 passages."""
    lookup = {(0, 0): "GOLD_PASSAGE"}
    for i in range(8):
        lookup[(1, i)] = f"HARD_{i}"
        lookup[(2, i)] = f"RAND_{i}"

    for pos in range(n + 1):
        row = _make_synthetic_input_row(
            qid="q_pos",
            gold_pid=[0, 0],
            gold_positions={"1": min(pos, 1), "2": min(pos, 2), "4": min(pos, 4), "8": min(pos, 8)},
        )
        row["gold_positions"][str(n)] = pos

        pids_hard = assemble_rag_passage_ids(row, "rag", "hard_negative", n)
        assert len(pids_hard) == n + 1
        assert pids_hard[pos] == (0, 0)
        non_gold_hard = [pid for idx, pid in enumerate(pids_hard) if idx != pos]
        assert non_gold_hard == [(1, i) for i in range(n)]

        rendered, msgs = render_unit_messages(
            row, "rag", "random_length_matched", n, paragraph_lookup=lookup
        )
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        expected_doc_line = f"[Document {pos + 1}] GOLD_PASSAGE\n"
        assert expected_doc_line in rendered


# ---------------------------------------------------------------------------
# 2. Extraction Rules Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_output", "expected_answer", "expected_valid"),
    [
        # Standard clean and labeled outputs
        ("Paris", "Paris", True),
        ("Answer: Paris", "Paris", True),
        ("answer:   Paris  ", "Paris", True),
        ("A: 1928", "1928", True),
        ("a: Alexander Fleming", "Alexander Fleming", True),
        # First non-empty line taken; later lines ignored
        ("\n\n  Answer: 42 \nExplanation: Because 6 times 7 is 42.", "42", True),
        # Quotes and trailing periods preserved (handled later by SQuAD norm)
        ('Answer: "the Treaty of Utrecht."', '"the Treaty of Utrecht."', True),
        # Empty / whitespace outputs -> invalid
        ("", "", False),
        ("   \n\t  \n ", "", False),
        ("Answer:   ", "", False),
        ("A:", "", False),
        # Word-count boundary: 10 words valid, 11 words invalid
        ("one two three four five six seven eight nine ten", "one two three four five six seven eight nine ten", True),
        (
            "one two three four five six seven eight nine ten eleven",
            "one two three four five six seven eight nine ten eleven",
            False,
        ),
        # All 12 forbidden starting phrases (case-insensitive) -> invalid
        ("The answer is Paris.", "The answer is Paris.", False),
        ("Answer: the answer is 1928", "the answer is 1928", False),
        ("Based on the passage, Fleming.", "Based on the passage, Fleming.", False),
        ("According to Document 1, oxygen.", "According to Document 1, oxygen.", False),
        ("The context states 1973.", "The context states 1973.", False),
        ("The passage mentions Normandy.", "The passage mentions Normandy.", False),
        ("The document says London.", "The document says London.", False),
        ("I cannot find the answer.", "I cannot find the answer.", False),
        ("I can't answer this question.", "I can't answer this question.", False),
        ("I don't know.", "I don't know.", False),
        ("There is no mention of this.", "There is no mention of this.", False),
        ("Not mentioned in the text.", "Not mentioned in the text.", False),
        ("Sorry, the context does not say.", "Sorry, the context does not say.", False),
    ],
)
def test_extraction_rules_edge_cases(
    raw_output: str,
    expected_answer: str,
    expected_valid: bool,
) -> None:
    """Verify configs/extraction_v1.yaml rules across labels, empty, word limits, and forbidden phrases."""
    cfg = load_extraction_config(REPO_ROOT / "configs" / "extraction_v1.yaml")
    extracted, is_valid = extract_answer(raw_output, extraction_cfg=cfg)
    assert extracted == expected_answer
    assert is_valid is expected_valid


# ---------------------------------------------------------------------------
# 3. Backend Determinism & Chat Template Tests
# ---------------------------------------------------------------------------


def test_stub_backend_determinism_and_thinking_disabled() -> None:
    """StubBackend must be deterministic and disable thinking for Qwen3."""
    qwen_stub = StubBackend("Qwen3-8B")
    llama_stub = StubBackend("Llama-3.1-8B-Instruct")

    msgs = build_messages("Question: What is 2+2?\nAnswer with the shortest possible answer, no explanation.")
    qwen_prompt = qwen_stub.format_prompt(msgs)
    llama_prompt = llama_stub.format_prompt(msgs)

    assert "enable_thinking=False" in qwen_prompt
    assert "enable_thinking" not in llama_prompt

    out1 = qwen_stub.generate([qwen_prompt, llama_prompt], max_new_tokens=32, greedy=True)
    out2 = qwen_stub.generate([qwen_prompt, llama_prompt], max_new_tokens=32, greedy=True)
    assert out1 == out2
    assert out1[0] != out1[1]

    # System prompt must be rejected
    with pytest.raises(ValueError, match="System prompt is forbidden"):
        qwen_stub.format_prompt([{"role": "system", "content": "sys"}, msgs[0]])


# ---------------------------------------------------------------------------
# 4. Seed-Independent vs Seed-Dependent Units, Resume, Schema, and Hashes
# ---------------------------------------------------------------------------


def test_seed_independent_and_distractor_units_and_row_schema(
    sandbox_repo: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify seed-independent union behavior, missing-pair skipping, and full row schema."""
    stub = StubBackend("Qwen3-8B")
    out_dir = sandbox_repo / "scratch" / "stub"

    # 1. Seed-independent oracle unit: pilot has 4 unique questions across seeds 0, 1, 2
    #    (even though pilot_q4 is missing in seed0 and pilot_q2 is missing in seed2)
    oracle_unit = parse_run_unit("Qwen3-8B__oracle__none__n0__seednone__pilot")
    res_oracle = run_unit(
        oracle_unit,
        stub,
        results_dir=out_dir,
        repo_root=sandbox_repo,
    )
    assert res_oracle["written_count"] == 4
    assert res_oracle["output_path"] == out_dir / "pilot" / oracle_unit.filename

    oracle_rows = [
        json.loads(line)
        for line in res_oracle["output_path"].read_text(encoding="utf-8").splitlines()
    ]
    assert [r["qid"] for r in oracle_rows] == ["pilot_q1", "pilot_q2", "pilot_q3", "pilot_q4"]

    expected_hashes = compute_study_hashes(repo_root=sandbox_repo)
    required_keys = {
        "qid",
        "model",
        "condition",
        "distractor_type",
        "n",
        "seed",
        "split",
        "backend",
        "is_stub",
        "model_id",
        "prompt_sha256",
        "raw_output",
        "extracted_answer",
        "valid",
        "seconds",
        "tokens_per_second",
        "config_hash",
        "prereg_hash",
        "manifest_hash",
    }
    for row in oracle_rows:
        assert set(row.keys()) == required_keys
        assert row["seed"] is None
        assert row["distractor_type"] is None
        assert row["n"] == 0
        assert row["backend"] == "stub"
        assert row["is_stub"] is True
        assert row["valid"] is True
        assert row["config_hash"] == expected_hashes["config_hash"]
        assert row["prereg_hash"] == expected_hashes["prereg_hash"]
        assert row["manifest_hash"] == expected_hashes["manifest_hash"]

    # 2. Seed-dependent distractor unit: seed 0 has 3 rows (pilot_q4 was excluded by P2c)
    hard_unit = parse_run_unit("Qwen3-8B__rag__hard_negative__n2__seed0__pilot")
    with caplog.at_level(logging.INFO):
        res_hard = run_unit(
            hard_unit,
            stub,
            results_dir=out_dir,
            repo_root=sandbox_repo,
        )
    assert res_hard["written_count"] == 3
    assert res_hard["skipped_p2c_count"] == 1
    assert any("skipped 1 (question, seed=0) pairs excluded by P2c" in m for m in caplog.messages)

    hard_rows = [
        json.loads(line)
        for line in res_hard["output_path"].read_text(encoding="utf-8").splitlines()
    ]
    assert [r["qid"] for r in hard_rows] == ["pilot_q1", "pilot_q2", "pilot_q3"]
    assert all(r["seed"] == 0 and r["distractor_type"] == "hard_negative" and r["n"] == 2 for r in hard_rows)


def test_resume_skips_existing_qids_without_rewriting(sandbox_repo: Path) -> None:
    """Verify resuming a partially completed unit skips existing qids and preserves prior bytes."""
    stub = StubBackend("Qwen3-8B")
    out_dir = sandbox_repo / "scratch" / "stub"
    unit = parse_run_unit("Qwen3-8B__closed_book__none__n0__seednone__main")

    # Step 1: Run first 2 questions
    res1 = run_unit(unit, stub, max_questions=2, results_dir=out_dir, repo_root=sandbox_repo)
    assert res1["written_count"] == 2
    assert res1["skipped_resume_count"] == 0
    initial_bytes = res1["output_path"].read_bytes()

    # Step 2: Re-run first 2 questions -> 0 new writes, identical file bytes
    res2 = run_unit(unit, stub, max_questions=2, results_dir=out_dir, repo_root=sandbox_repo)
    assert res2["written_count"] == 0
    assert res2["skipped_resume_count"] == 2
    assert res2["output_path"].read_bytes() == initial_bytes

    # Step 3: Resume up to all 4 questions -> 2 skipped, 2 new rows appended
    res3 = run_unit(unit, stub, max_questions=4, results_dir=out_dir, repo_root=sandbox_repo)
    assert res3["written_count"] == 2
    assert res3["skipped_resume_count"] == 2
    final_bytes = res3["output_path"].read_bytes()
    assert final_bytes.startswith(initial_bytes)

    rows = [json.loads(line) for line in final_bytes.decode("utf-8").splitlines()]
    assert [r["qid"] for r in rows] == ["main_q1", "main_q2", "main_q3", "main_q4"]


# ---------------------------------------------------------------------------
# 5. Guardrails: Stub & Dry-Run Refuse results/, Official Updates LEDGER.csv
# ---------------------------------------------------------------------------


def test_stub_and_dry_run_refuse_results_directory(sandbox_repo: Path) -> None:
    """Verify --backend stub and --dry-run force scratch/ and refuse any path under results/."""
    assert build_arg_parser().get_default("results_dir") == "results"

    # Default directory resolution when requested_results_dir is None
    assert resolve_results_dir(
        None, backend_name="stub", is_stub=True, dry_run=False, repo_root=sandbox_repo
    ) == (sandbox_repo / "scratch" / "stub").resolve()
    assert resolve_results_dir(
        None, backend_name="mlx", is_stub=False, dry_run=True, repo_root=sandbox_repo
    ) == (sandbox_repo / "scratch" / "dryrun").resolve()
    assert resolve_results_dir(
        None, backend_name="mlx", is_stub=False, dry_run=False, repo_root=sandbox_repo
    ) == (sandbox_repo / "results").resolve()

    # Stub refuses results/ and results/pilot/
    for forbidden in (
        "results",
        "results/pilot",
        sandbox_repo / "results",
        sandbox_repo / "results" / "pilot",
        REPO_ROOT / "results",
        REPO_ROOT / "results" / "pilot",
    ):
        with pytest.raises(ValueError, match="Refusing to write stub or dry-run outputs under results/"):
            resolve_results_dir(
                forbidden,
                backend_name="stub",
                is_stub=True,
                dry_run=False,
                repo_root=sandbox_repo,
            )
        with pytest.raises(ValueError, match="Refusing to write stub or dry-run outputs under results/"):
            resolve_results_dir(
                forbidden,
                backend_name="mlx",
                is_stub=False,
                dry_run=True,
                repo_root=sandbox_repo,
            )

    # CLI refuses --backend stub --results-dir results
    rc_stub = run_units_main(
        [
            "--units",
            "Qwen3-8B__oracle__none__n0__seednone__pilot",
            "--questions",
            "2",
            "--backend",
            "stub",
            "--results-dir",
            "results",
            "--repo-root",
            str(sandbox_repo),
        ]
    )
    assert rc_stub != 0
    assert not (sandbox_repo / "results").exists()

    # CLI refuses --dry-run --results-dir results/pilot
    rc_dry = run_units_main(
        [
            "--units",
            "Qwen3-8B__oracle__none__n0__seednone__pilot",
            "--questions",
            "2",
            "--dry-run",
            "--results-dir",
            "results/pilot",
            "--repo-root",
            str(sandbox_repo),
        ]
    )
    assert rc_dry != 0
    assert not (sandbox_repo / "results").exists()

    # CLI with --backend stub (no --results-dir) writes to sandbox_repo/scratch/stub/pilot/
    rc_ok = run_units_main(
        [
            "--units",
            "Qwen3-8B__oracle__none__n0__seednone__pilot",
            "--questions",
            "2",
            "--backend",
            "stub",
            "--repo-root",
            str(sandbox_repo),
        ]
    )
    assert rc_ok == 0
    expected_stub_file = (
        sandbox_repo
        / "scratch"
        / "stub"
        / "pilot"
        / "Qwen3-8B__oracle__none__n0__seednone__pilot.jsonl"
    )
    assert expected_stub_file.exists()
    assert not (sandbox_repo / "results").exists()


def test_official_run_writes_ledger_and_passes_invariants(sandbox_repo: Path) -> None:
    """Verify official runs write pilot units to results/pilot/, append to LEDGER.csv, and pass check_invariants."""
    official_backend = FakeOfficialBackend("Qwen3-8B")
    pilot_unit = parse_run_unit("Qwen3-8B__oracle__none__n0__seednone__pilot")
    main_unit = parse_run_unit("Qwen3-8B__rag__hard_negative__n2__seed0__main")

    # Run first 2 questions of pilot unit
    r1 = run_unit(pilot_unit, official_backend, max_questions=2, repo_root=sandbox_repo)
    assert r1["output_path"] == sandbox_repo / "results" / "pilot" / pilot_unit.filename
    assert r1["ledger_entry"] is not None

    # Resume pilot unit for all 4 questions (appends 2nd entry to LEDGER.csv)
    r2 = run_unit(pilot_unit, official_backend, max_questions=4, repo_root=sandbox_repo)
    assert r2["written_count"] == 2
    assert r2["ledger_entry"] is not None

    # Run main unit
    r3 = run_unit(main_unit, official_backend, max_questions=3, repo_root=sandbox_repo)
    assert r3["output_path"] == sandbox_repo / "results" / main_unit.filename

    ledger_file = sandbox_repo / "results" / "LEDGER.csv"
    assert ledger_file.exists()
    with ledger_file.open("r", encoding="utf-8") as f:
        ledger_rows = list(csv.DictReader(f))
    assert len(ledger_rows) == 3
    assert ledger_rows[0]["file"] == f"results/pilot/{pilot_unit.filename}"
    assert ledger_rows[0]["line_count"] == "2"
    assert ledger_rows[1]["file"] == f"results/pilot/{pilot_unit.filename}"
    assert ledger_rows[1]["line_count"] == "4"
    assert ledger_rows[2]["file"] == f"results/{main_unit.filename}"
    assert ledger_rows[2]["line_count"] == "3"

    # Invariants check must pass on the sandbox repository
    errors = check_invariants(repo_root=sandbox_repo, run_env_check=False)
    assert errors == []

    # Analysis loader loads all 7 official rows cleanly
    loaded = load_result_rows(sandbox_repo / "results", repo_root=sandbox_repo)
    assert len(loaded) == 7


def test_check_invariants_and_loader_reject_is_stub_under_results(sandbox_repo: Path) -> None:
    """Verify check_invariants and load_result_rows flag any is_stub=True row under results/ recursively."""
    nested_dir = sandbox_repo / "results" / "pilot"
    nested_dir.mkdir(parents=True, exist_ok=True)
    bad_file = nested_dir / "Qwen3-8B__oracle__none__n0__seednone__pilot.jsonl"

    hashes = compute_study_hashes(repo_root=sandbox_repo)
    bad_row = {
        "qid": "pilot_q1",
        "model": "Qwen3-8B",
        "condition": "oracle",
        "distractor_type": None,
        "n": 0,
        "seed": None,
        "split": "pilot",
        "backend": "stub",
        "is_stub": True,
        "model_id": "stub/Qwen3-8B",
        "prompt_sha256": "abc",
        "raw_output": "Answer: stub_123",
        "extracted_answer": "stub_123",
        "valid": True,
        "seconds": 0.001,
        "tokens_per_second": 1000.0,
        "config_hash": hashes["config_hash"],
        "prereg_hash": hashes["prereg_hash"],
        "manifest_hash": hashes["manifest_hash"],
    }
    bad_file.write_text(json.dumps(bad_row) + "\n", encoding="utf-8")

    # 1. check_invariants must catch the stub row in results/pilot/
    errors = check_invariants(repo_root=sandbox_repo, run_env_check=False)
    assert any("is_stub=True" in e and "pilot/" in e for e in errors), (
        f"Expected recursive is_stub error in {errors}"
    )

    # 2. Analysis loader must also reject the stub row under results/
    with pytest.raises(ValueError, match="stub row \\(is_stub=True\\) is forbidden"):
        load_result_rows(sandbox_repo / "results", repo_root=sandbox_repo)


def test_frozen_main_inputs_and_pilot_manifest_hash_separation(sandbox_repo: Path) -> None:
    """When data/frozen/manifest.json exists, main uses data/frozen/ while pilot uses data/sampling/."""
    frozen_dir = sandbox_repo / "data" / "frozen"
    frozen_dir.mkdir(parents=True, exist_ok=True)

    frozen_qids = ["frozen_q1", "frozen_q2", "frozen_q3"]
    (frozen_dir / "main.json").write_text(
        json.dumps([{"question_id": qid} for qid in frozen_qids]),
        encoding="utf-8",
    )
    for seed in (0, 1, 2):
        rows = [_make_synthetic_input_row(qid, split="main", seed=seed) for qid in frozen_qids]
        (frozen_dir / f"main_seed{seed}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows),
            encoding="utf-8",
        )

    frozen_manifest = frozen_dir / "manifest.json"
    frozen_manifest.write_text(
        json.dumps({"final_n": 3, "question_ids": frozen_qids}),
        encoding="utf-8",
    )
    frozen_hash = sha256_bytes(frozen_manifest.read_bytes())
    (frozen_dir / "manifest.sha256").write_text(
        f"{frozen_hash}  data/frozen/manifest.json\n",
        encoding="utf-8",
    )
    sampling_hash = sha256_bytes((sandbox_repo / "data" / "sampling" / "manifest.json").read_bytes())
    assert frozen_hash != sampling_hash

    official_backend = FakeOfficialBackend("Qwen3-8B")

    # 1. Seed-dependent main unit reads from data/frozen/ and logs frozen manifest_hash
    main_rag = parse_run_unit("Qwen3-8B__rag__hard_negative__n2__seed0__main")
    res_main_rag = run_unit(main_rag, official_backend, repo_root=sandbox_repo)
    assert res_main_rag["written_count"] == 3
    assert res_main_rag["skipped_p2c_count"] == 0
    main_rag_rows = [
        json.loads(line)
        for line in res_main_rag["output_path"].read_text(encoding="utf-8").splitlines()
    ]
    assert [r["qid"] for r in main_rag_rows] == frozen_qids
    assert all(r["manifest_hash"] == frozen_hash for r in main_rag_rows)

    # 2. Seed-independent main unit reads from data/frozen/ and logs frozen manifest_hash
    main_oracle = parse_run_unit("Qwen3-8B__oracle__none__n0__seednone__main")
    res_main_oracle = run_unit(main_oracle, official_backend, repo_root=sandbox_repo)
    main_oracle_rows = [
        json.loads(line)
        for line in res_main_oracle["output_path"].read_text(encoding="utf-8").splitlines()
    ]
    assert [r["qid"] for r in main_oracle_rows] == frozen_qids
    assert all(r["manifest_hash"] == frozen_hash for r in main_oracle_rows)

    # 3. Pilot unit continues reading from data/sampling/ + data/inputs/ and logs sampling manifest_hash
    pilot_oracle = parse_run_unit("Qwen3-8B__oracle__none__n0__seednone__pilot")
    res_pilot = run_unit(pilot_oracle, official_backend, repo_root=sandbox_repo)
    pilot_rows = [
        json.loads(line)
        for line in res_pilot["output_path"].read_text(encoding="utf-8").splitlines()
    ]
    assert [r["qid"] for r in pilot_rows] == ["pilot_q1", "pilot_q2", "pilot_q3", "pilot_q4"]
    assert all(r["manifest_hash"] == sampling_hash for r in pilot_rows)

    # 4. Invariants check passes with both pilot (sampling hash) and main (frozen hash) rows
    assert check_invariants(repo_root=sandbox_repo, run_env_check=False) == []

    # 5. Explicit inputs_dir overrides data/frozen/ for custom fixtures
    stub = StubBackend("Qwen3-8B")
    res_custom = run_unit(
        main_rag,
        stub,
        inputs_dir=sandbox_repo / "data" / "inputs",
        repo_root=sandbox_repo,
    )
    custom_rows = [
        json.loads(line)
        for line in res_custom["output_path"].read_text(encoding="utf-8").splitlines()
    ]
    assert [r["qid"] for r in custom_rows] == ["main_q1", "main_q2", "main_q3"]

