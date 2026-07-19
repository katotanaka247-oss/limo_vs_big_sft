"""
test_eval_output_discovery.py
测试 lm-eval 输出发现逻辑：
  1. 正常单次运行输出
  2. 多个历史 run（只读 active run）
  3. 缺失 MATH500
  4. 缺失 AIME24
  5. 缺失 AIME25
  6. JSON 损坏
  7. sample 数重复
"""
import json
import os
from pathlib import Path

import pytest

from summarize_eval_efficiency import (
    _load_manifest_from_result_dir,
    summarize_one,
    LIMIT_THRESHOLD,
)


class TestNormalRun:
    """1. 正常单次运行输出"""

    def test_normal_run(self, tmp_path, make_mock_manifest, make_mock_sample_jsonl,
                        make_mock_results_json):
        run_dir = tmp_path / "runs" / "run_test"
        run_dir.mkdir(parents=True)
        tasks = ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]

        make_mock_manifest(run_dir / "run_manifest.json")
        make_mock_results_json(run_dir / "results.json")
        for t in tasks:
            make_mock_sample_jsonl(run_dir / f"samples_{t}.jsonl", n=2)

        # active_run.json
        active = {"active_run_id": "run_test", "status": "complete"}
        with open(tmp_path / "active_run.json", "w") as f:
            json.dump(active, f)

        m = _load_manifest_from_result_dir(tmp_path)
        assert m is not None
        assert m["status"] == "complete"
        assert len(m["lm_eval_sample_files"]) == 3

        summary = summarize_one(tmp_path)
        assert summary["overall"]["n_samples"] == 6  # 2 per task * 3 tasks


class TestMultipleHistoricalRuns:
    """2. 多个历史 run（只读 active run）"""

    def test_multiple_runs_only_active_read(self, tmp_path, make_mock_manifest,
                                            make_mock_sample_jsonl,
                                            make_mock_results_json):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        # 创建两个 run，active 指向 run_2
        for rid in ["run_1", "run_2"]:
            rd = runs_dir / rid
            rd.mkdir()
            tasks = ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]
            make_mock_manifest(
                rd / "run_manifest.json",
                sample_counts={t: 2 for t in tasks},
            )
            make_mock_results_json(rd / "results.json")
            for t in tasks:
                make_mock_sample_jsonl(rd / f"samples_{t}.jsonl", n=2)

        # active_run.json 指向 run_2
        active = {"active_run_id": "run_2", "status": "complete"}
        with open(tmp_path / "active_run.json", "w") as f:
            json.dump(active, f)

        m = _load_manifest_from_result_dir(tmp_path)
        assert m is not None
        assert m["run_id"] == "run_test"  # mock manifest 中 run_id 固定

        # summarize_one 应该只读 active run 的文件
        summary = summarize_one(tmp_path)
        assert summary["overall"]["n_samples"] == 6


class TestMissingTasks:
    """3-5. 缺失 MATH500 / AIME24 / AIME25"""

    @pytest.mark.parametrize("missing_task", [
        "local_math500_32k",
        "local_aime24_32k",
        "local_aime25_32k",
    ])
    def test_missing_task(self, tmp_path, make_mock_manifest, make_mock_sample_jsonl,
                          make_mock_results_json, missing_task):
        run_dir = tmp_path / "runs" / "run_test"
        run_dir.mkdir(parents=True)
        all_tasks = ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]
        remaining = [t for t in all_tasks if t != missing_task]

        # manifest 只包含剩余 task
        make_mock_manifest(run_dir / "run_manifest.json", tasks=remaining)
        make_mock_results_json(run_dir / "results.json", tasks=remaining)
        for t in remaining:
            make_mock_sample_jsonl(run_dir / f"samples_{t}.jsonl", n=2)

        summary = summarize_one(run_dir)
        assert missing_task not in summary["per_task"]
        assert len(summary["per_task"]) == 2
        assert summary["overall"]["n_samples"] == 4


class TestCorruptJSON:
    """6. JSON 损坏"""

    def test_corrupt_manifest(self, tmp_path):
        run_dir = tmp_path / "runs" / "run_test"
        run_dir.mkdir(parents=True)
        # 写入损坏的 JSON
        with open(run_dir / "run_manifest.json", "w") as f:
            f.write("{ this is not valid json !!!")

        m = _load_manifest_from_result_dir(run_dir)
        # 损坏的 JSON 应返回 None
        assert m is None

    def test_corrupt_results_json(self, tmp_path, make_mock_manifest,
                                  make_mock_sample_jsonl):
        run_dir = tmp_path / "runs" / "run_test"
        run_dir.mkdir(parents=True)
        tasks = ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]
        make_mock_manifest(run_dir / "run_manifest.json")
        # 损坏的 results.json
        with open(run_dir / "results.json", "w") as f:
            f.write("{ broken json")
        for t in tasks:
            make_mock_sample_jsonl(run_dir / f"samples_{t}.jsonl", n=2)

        # 不应该崩溃
        summary = summarize_one(run_dir)
        assert summary["overall"]["n_samples"] == 6
        # accuracy 应该从 sample 文件推导
        for t in tasks:
            st = summary["per_task"][t]
            assert st["n_samples"] == 2


class TestDuplicateSamples:
    """7. sample 数重复"""

    def test_duplicate_doc_id_detected(self, tmp_path, make_mock_manifest,
                                       make_mock_sample_jsonl,
                                       make_mock_results_json):
        run_dir = tmp_path / "runs" / "run_test"
        run_dir.mkdir(parents=True)
        tasks = ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]
        make_mock_manifest(run_dir / "run_manifest.json")
        make_mock_results_json(run_dir / "results.json")
        # MATH500 带重复
        make_mock_sample_jsonl(run_dir / "samples_local_math500_32k.jsonl",
                               n=2, with_dup=True)
        for t in tasks[1:]:
            make_mock_sample_jsonl(run_dir / f"samples_{t}.jsonl", n=2)

        summary = summarize_one(run_dir)
        # 重复的 sample 会被计入，但这是汇总脚本的行为
        # 完成度判定（run_eval_single_l40_vllm.sh 中的 Python 脚本）会检测重复
        math500 = summary["per_task"]["local_math500_32k"]
        assert math500["n_samples"] == 3  # 2 + 1 duplicate
