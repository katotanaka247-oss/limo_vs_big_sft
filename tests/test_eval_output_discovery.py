"""
test_eval_output_discovery.py
测试 lm-eval 输出发现逻辑（generation-only 模式）：
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
    _analyze_model,
)


def _create_run(model_dir, run_id="run_test", tasks=None, n_samples=2,
                status="complete", with_duplicates=False):
    """手动创建一个完整的 run 目录结构。"""
    tasks = tasks or ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # active_run.json
    with open(model_dir / "active_run.json", "w") as f:
        json.dump({"active_run_id": run_id, "status": status}, f)

    run_dir = model_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    sample_files = {}
    for task in tasks:
        task_dir = run_dir / "tasks" / task
        task_dir.mkdir(parents=True, exist_ok=True)
        sample_path = task_dir / "samples.jsonl"
        with open(sample_path, "w") as f:
            for i in range(n_samples):
                f.write(json.dumps({
                    "doc_id": i,
                    "doc": {"problem": f"Problem {i}", "answer": str(i)},
                    "arguments": [[f"Problem: Problem {i}\nAnswer:"]],
                    "resps": [[f"answer {i}"]],
                    "filtered_resps": [[f"answer {i}"]],
                }) + "\n")
            if with_duplicates and task == "local_math500_32k":
                f.write(json.dumps({
                    "doc_id": 0,
                    "doc": {"problem": "Problem 0", "answer": "0"},
                    "arguments": [["Problem: Problem 0\nAnswer:"]],
                    "resps": [["duplicate"]],
                    "filtered_resps": [["duplicate"]],
                }) + "\n")
        sample_files[task] = str(sample_path)

        # task manifest
        tm = {
            "status": status, "task": task,
            "model_name": "test_model",
            "lm_eval_sample_file": str(sample_path),
            "expected_sample_count": n_samples,
            "actual_sample_count": n_samples + (1 if with_duplicates and task == "local_math500_32k" else 0),
            "max_gen_toks": 32768,
            "max_model_len": 40960,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 32,
            "gpu_memory_utilization": 0.90,
            "enable_prefix_caching": True,
            "successful_attempt_elapsed_seconds": 1800,
        }
        with open(task_dir / "task_manifest.json", "w") as f:
            json.dump(tm, f)

    # run manifest
    actual_counts = {t: n_samples + (1 if with_duplicates and t == "local_math500_32k" else 0)
                     for t in tasks}
    rm = {
        "status": status,
        "model_name": "test_model",
        "model_path": "/fake/model",
        "run_id": run_id,
        "tasks": tasks,
        "backend": "vllm",
        "evaluation_mode": "generation_only",
        "predict_only": True,
        "judging_status": "pending_local",
        "max_gen_toks": 32768,
        "max_model_len": 40960,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 32,
        "gpu_memory_utilization": 0.90,
        "enable_prefix_caching": True,
        "fallback_level": 1,
        "successful_attempt_elapsed_seconds": 1800,
        "failed_attempt_elapsed_seconds": 0,
        "pipeline_elapsed_seconds": 1800,
        "gpu_peak_memory_mib": 42120,
        "vllm_version": "0.6.6.post1",
        "lm_eval_version": "0.4.5",
        "dtype": "bfloat16",
        "expected_sample_counts": {t: n_samples for t in tasks},
        "actual_sample_counts": actual_counts,
    }
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(rm, f)

    return model_dir


class TestNormalRun:
    """1. 正常单次运行输出"""

    def test_normal_run(self, tmp_path):
        model_dir = _create_run(tmp_path / "results" / "limo", n_samples=2)

        manifest, run_dir = _load_manifest_from_result_dir(str(model_dir))
        assert manifest is not None
        assert manifest["status"] == "complete"

        stats = _analyze_model(str(model_dir), "LIMO-817")
        assert stats["total_actual_samples"] == 6  # 2 per task * 3 tasks
        assert stats["total_expected_samples"] == 6
        assert len(stats["task_stats"]) == 3
        assert stats["generation_complete"] is True


class TestMultipleHistoricalRuns:
    """2. 多个历史 run（只读 active run）"""

    def test_multiple_runs_only_active_read(self, tmp_path):
        model_dir = tmp_path / "results" / "limo"
        # 创建两个 run，active 指向 run_2
        _create_run(model_dir, run_id="run_1", n_samples=2)
        _create_run(model_dir, run_id="run_2", n_samples=3)

        # active_run.json 指向 run_2
        with open(model_dir / "active_run.json", "w") as f:
            json.dump({"active_run_id": "run_2", "status": "complete"}, f)

        manifest, run_dir = _load_manifest_from_result_dir(str(model_dir))
        assert manifest["run_id"] == "run_2"

        # _analyze_model 应该只读 active run 的文件
        stats = _analyze_model(str(model_dir), "LIMO-817")
        assert stats["total_actual_samples"] == 9  # 3 per task * 3 tasks


class TestMissingTasks:
    """3-5. 缺失 MATH500 / AIME24 / AIME25"""

    @pytest.mark.parametrize("missing_task", [
        "local_math500_32k",
        "local_aime24_32k",
        "local_aime25_32k",
    ])
    def test_missing_task(self, tmp_path, missing_task):
        all_tasks = ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]
        remaining = [t for t in all_tasks if t != missing_task]
        model_dir = _create_run(tmp_path / "results" / "limo",
                                tasks=remaining, n_samples=2)

        stats = _analyze_model(str(model_dir), "LIMO-817")
        assert missing_task not in stats["task_stats"]
        assert len(stats["task_stats"]) == 2
        assert stats["total_actual_samples"] == 4


class TestCorruptJSON:
    """6. JSON 损坏"""

    def test_corrupt_manifest_raises(self, tmp_path):
        """损坏的 run_manifest.json 应导致 _load_manifest_from_result_dir 抛出异常"""
        model_dir = tmp_path / "results" / "limo"
        run_dir = model_dir / "runs" / "run_test"
        run_dir.mkdir(parents=True)
        with open(run_dir / "run_manifest.json", "w") as f:
            f.write("{ this is not valid json !!!")
        with open(model_dir / "active_run.json", "w") as f:
            json.dump({"active_run_id": "run_test", "status": "complete"}, f)

        with pytest.raises(Exception):
            _load_manifest_from_result_dir(str(model_dir))

    def test_corrupt_manifest_handled_by_analyze(self, tmp_path):
        """_analyze_model 应捕获损坏的 manifest 并返回 error"""
        model_dir = tmp_path / "results" / "limo"
        run_dir = model_dir / "runs" / "run_test"
        run_dir.mkdir(parents=True)
        with open(run_dir / "run_manifest.json", "w") as f:
            f.write("{ broken json")
        with open(model_dir / "active_run.json", "w") as f:
            json.dump({"active_run_id": "run_test", "status": "complete"}, f)

        stats = _analyze_model(str(model_dir), "LIMO-817")
        assert "error" in stats
        assert stats["generation_complete"] is False


class TestDuplicateSamples:
    """7. sample 数重复"""

    def test_duplicate_doc_id_detected(self, tmp_path):
        model_dir = _create_run(tmp_path / "results" / "limo",
                                n_samples=2, with_duplicates=True)

        stats = _analyze_model(str(model_dir), "LIMO-817")
        math500 = stats["task_stats"]["local_math500_32k"]
        # 2 original + 1 duplicate = 3
        assert math500["actual_samples"] == 3
        assert math500["duplicate_count"] == 1

    def test_no_duplicates_in_normal_run(self, tmp_path):
        model_dir = _create_run(tmp_path / "results" / "limo", n_samples=2)

        stats = _analyze_model(str(model_dir), "LIMO-817")
        for task, ts in stats["task_stats"].items():
            assert ts["duplicate_count"] == 0


class TestManifestBasedReading:
    """manifest 精确读取，不递归读取历史文件"""

    def test_only_manifest_files_read(self, tmp_path):
        model_dir = _create_run(tmp_path / "results" / "limo", n_samples=2)
        run_dir = model_dir / "runs" / "run_test"

        # 创建一个无关的 JSONL 文件（不应该被读取）
        with open(run_dir / "random_old_samples.jsonl", "w") as f:
            f.write(json.dumps({"doc_id": "old", "resps": [["old"]]}) + "\n")

        stats = _analyze_model(str(model_dir), "LIMO-817")
        # 只读 manifest 指定的 3 个文件，每文件 2 条 = 6 条
        assert stats["total_actual_samples"] == 6
