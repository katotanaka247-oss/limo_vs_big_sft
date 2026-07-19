"""
test_eval_summary.py
测试效率统计逻辑：
  * manifest 精确读取
  * finish_reason 始终为 "unknown"
  * truncation 阈值判定
  * P50/P90/P95/percentile 正确性
  * 两模型配置一致性检查
"""
import json
from pathlib import Path

import pytest

from summarize_eval_efficiency import (
    summarize_one,
    write_comparison,
    _percentile,
    LIMIT_THRESHOLD,
    MAX_GEN_TOKS,
)


class TestFinishReason:
    """finish_reason 必须始终为 'unknown'"""

    def test_finish_reason_unknown(self, tmp_path, make_mock_manifest,
                                   make_mock_sample_jsonl, make_mock_results_json):
        run_dir = tmp_path / "runs" / "run_test"
        run_dir.mkdir(parents=True)
        tasks = ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]
        make_mock_manifest(run_dir / "run_manifest.json")
        make_mock_results_json(run_dir / "results.json")
        for t in tasks:
            make_mock_sample_jsonl(run_dir / f"samples_{t}.jsonl", n=2)

        summary = summarize_one(run_dir)
        for t in tasks:
            st = summary["per_task"][t]
            assert st["finish_reason_unknown_ratio"] == 1.0


class TestTruncationThreshold:
    """truncation 只根据 token 数判断"""

    def test_truncation_at_threshold(self, tmp_path, make_mock_manifest,
                                     make_mock_sample_jsonl, make_mock_results_json):
        run_dir = tmp_path / "runs" / "run_test"
        run_dir.mkdir(parents=True)
        tasks = ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]
        make_mock_manifest(run_dir / "run_manifest.json")
        make_mock_results_json(run_dir / "results.json")
        # MATH500 中一个样本 token 数 >= LIMIT_THRESHOLD
        make_mock_sample_jsonl(
            run_dir / "samples_local_math500_32k.jsonl",
            n=2, token_counts=[100, LIMIT_THRESHOLD + 10],
        )
        for t in tasks[1:]:
            make_mock_sample_jsonl(run_dir / f"samples_{t}.jsonl", n=2)

        summary = summarize_one(run_dir)
        math500 = summary["per_task"]["local_math500_32k"]
        assert math500["reached_limit_count"] == 1
        assert math500["truncation_rate"] == 0.5

    def test_no_truncation_below_threshold(self, tmp_path, make_mock_manifest,
                                           make_mock_sample_jsonl,
                                           make_mock_results_json):
        run_dir = tmp_path / "runs" / "run_test"
        run_dir.mkdir(parents=True)
        tasks = ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]
        make_mock_manifest(run_dir / "run_manifest.json")
        make_mock_results_json(run_dir / "results.json")
        for t in tasks:
            make_mock_sample_jsonl(run_dir / f"samples_{t}.jsonl",
                                   n=2, token_counts=[100, 200])

        summary = summarize_one(run_dir)
        for t in tasks:
            assert summary["per_task"][t]["reached_limit_count"] == 0
            assert summary["per_task"][t]["truncation_rate"] == 0.0


class TestPercentile:
    """P50/P90/P95 正确性"""

    def test_percentile_basic(self):
        vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        assert _percentile(vals, 0.50) == 55.0  # 插值
        assert _percentile(vals, 0.90) >= 90
        assert _percentile(vals, 0.95) >= 90

    def test_percentile_single(self):
        assert _percentile([42], 0.50) == 42

    def test_percentile_empty(self):
        assert _percentile([], 0.50) == 0


class TestConfigComparison:
    """两模型配置一致性检查"""

    def test_configs_match(self, tmp_path, make_mock_manifest, make_mock_sample_jsonl,
                           make_mock_results_json):
        limo_dir = tmp_path / "limo" / "runs" / "run_1"
        openr1_dir = tmp_path / "openr1" / "runs" / "run_1"
        limo_dir.mkdir(parents=True)
        openr1_dir.mkdir(parents=True)
        tasks = ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]

        for d in [limo_dir, openr1_dir]:
            make_mock_manifest(d / "run_manifest.json")
            make_mock_results_json(d / "results.json")
            for t in tasks:
                make_mock_sample_jsonl(d / f"samples_{t}.jsonl", n=2)

        limo_sum = summarize_one(limo_dir)
        openr1_sum = summarize_one(openr1_dir)
        out_json = str(tmp_path / "comparison.json")
        out_csv = str(tmp_path / "comparison.csv")
        out_md = str(tmp_path / "comparison.md")

        write_comparison(limo_sum, openr1_sum, out_json, out_csv, out_md,
                         throughput_comparable=True)

        with open(out_json, encoding="utf-8") as f:
            comp = json.load(f)
        assert comp["config_match"] is True
        assert comp["throughput_comparable"] is True
        assert comp["accuracy_comparable"] is True

    def test_configs_mismatch(self, tmp_path, make_mock_manifest,
                              make_mock_sample_jsonl, make_mock_results_json):
        limo_dir = tmp_path / "limo" / "runs" / "run_1"
        openr1_dir = tmp_path / "openr1" / "runs" / "run_1"
        limo_dir.mkdir(parents=True)
        openr1_dir.mkdir(parents=True)
        tasks = ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]

        # LIMO: max_num_seqs=32
        make_mock_manifest(limo_dir / "run_manifest.json", max_num_seqs=32)
        make_mock_results_json(limo_dir / "results.json")
        for t in tasks:
            make_mock_sample_jsonl(limo_dir / f"samples_{t}.jsonl", n=2)

        # OpenR1: max_num_seqs=8（更保守）
        make_mock_manifest(openr1_dir / "run_manifest.json", max_num_seqs=8)
        make_mock_results_json(openr1_dir / "results.json")
        for t in tasks:
            make_mock_sample_jsonl(openr1_dir / f"samples_{t}.jsonl", n=2)

        limo_sum = summarize_one(limo_dir)
        openr1_sum = summarize_one(openr1_dir)
        out_json = str(tmp_path / "comparison.json")
        out_csv = str(tmp_path / "comparison.csv")
        out_md = str(tmp_path / "comparison.md")

        write_comparison(limo_sum, openr1_sum, out_json, out_csv, out_md,
                         throughput_comparable=False)

        with open(out_json, encoding="utf-8") as f:
            comp = json.load(f)
        assert comp["config_match"] is False
        assert comp["throughput_comparable"] is False


class TestManifestBasedReading:
    """manifest 精确读取，不递归读取历史文件"""

    def test_only_manifest_files_read(self, tmp_path, make_mock_manifest,
                                      make_mock_sample_jsonl,
                                      make_mock_results_json):
        run_dir = tmp_path / "runs" / "run_test"
        run_dir.mkdir(parents=True)
        tasks = ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]
        make_mock_manifest(run_dir / "run_manifest.json")
        make_mock_results_json(run_dir / "results.json")
        for t in tasks:
            make_mock_sample_jsonl(run_dir / f"samples_{t}.jsonl", n=2)

        # 创建一个无关的 JSONL 文件（不应该被读取）
        with open(run_dir / "random_old_samples.jsonl", "w") as f:
            f.write(json.dumps({"doc_id": "old", "resps": [["old"]]}) + "\n")

        summary = summarize_one(run_dir)
        # 只读 manifest 指定的 3 个文件，每文件 2 条 = 6 条
        assert summary["overall"]["n_samples"] == 6
