"""conftest.py — 为 tests/ 提供 sys.path 注入与通用 fixture。"""
import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
sys.path.insert(0, str(PROJECT_DIR / "eval_tasks"))


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """提供一个临时项目目录，含 scripts/ 和 eval_tasks/ 子目录。"""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "eval_tasks").mkdir()
    (tmp_path / "results").mkdir()
    return tmp_path


@pytest.fixture
def make_mock_run_manifest():
    """生成一个 mock run_manifest.json 并写入指定路径。
    支持 generation-only 模式的新字段。"""
    def _make(path, status="complete", tasks=None,
              sample_counts=None, expected_counts=None,
              max_gen_toks=32768, max_model_len=40960,
              mnbt=8192, mns=32, gmu=0.90, epc=True,
              success_elapsed=1800, failed_elapsed=400,
              gpu_peak=42120, fallback_level=1,
              **overrides):
        tasks = tasks or ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]
        path = Path(path)
        sample_files = {}
        for t in tasks:
            sample_files[t] = str(path.parent / "tasks" / t / "samples.jsonl")
        manifest = {
            "status": status,
            "model_name": "test_model",
            "model_path": "/fake/model",
            "base_model": "meta-llama/Llama-3.1-8B",
            "run_id": "run_test",
            "tasks": tasks,
            "backend": "vllm",
            "evaluation_mode": "generation_only",
            "predict_only": True,
            "judging_status": "pending_local",
            "server_side_accuracy_valid": False,
            "dtype": "bfloat16",
            "temperature": 0.0,
            "do_sample": False,
            "max_gen_toks": max_gen_toks,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": mnbt,
            "max_num_seqs": mns,
            "gpu_memory_utilization": gmu,
            "enable_prefix_caching": epc,
            "fallback_level": fallback_level,
            "tensor_parallel_size": 1,
            "cuda_visible_devices": "0",
            "evaluation_protocol": "stock_zero_shot",
            "num_fewshot": 0,
            "apply_chat_template": False,
            "boxed_answer_instruction": False,
            "gpu_name": "NVIDIA L40",
            "gpu_uuid": "GPU-xxxx",
            "gpu_total_memory": "46068 MiB",
            "gpu_peak_memory_mib": gpu_peak,
            "lm_eval_version": "0.4.5",
            "vllm_version": "0.6.6.post1",
            "transformers_version": "4.46.3",
            "torch_version": "2.5.1+cu121",
            "peft_version": "0.13.2",
            "accelerate_version": "1.1.1",
            "datasets_version": "2.20.0",
            "git_commit": "test",
            "start_time": "2026-07-19 10:00:00",
            "end_time": "2026-07-19 11:00:00",
            "pipeline_elapsed_seconds": success_elapsed + failed_elapsed,
            "successful_attempt_elapsed_seconds": success_elapsed,
            "failed_attempt_elapsed_seconds": failed_elapsed,
            "successful_attempt": 1,
            "expected_sample_counts": expected_counts or {t: 2 for t in tasks},
            "actual_sample_counts": sample_counts or {t: 2 for t in tasks},
            "task_manifests": {t: f"tasks/{t}/task_manifest.json" for t in tasks},
            "lm_eval_results_file": str(path.parent / "results.json"),
            "lm_eval_sample_files": sample_files,
            "completion_errors": [],
        }
        manifest.update(overrides)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return manifest

    return _make


# 向后兼容：保留旧名称
@pytest.fixture
def make_mock_manifest(make_mock_run_manifest):
    """向后兼容的 manifest fixture。"""
    return make_mock_run_manifest


@pytest.fixture
def make_mock_task_manifest():
    """生成一个 mock task_manifest.json。"""
    def _make(path, task="local_math500_32k", status="complete",
              actual_count=2, expected_count=2,
              max_gen_toks=32768, max_model_len=40960,
              mnbt=8192, mns=32, gmu=0.90, epc=True,
              success_elapsed=1800, fallback_level=1,
              **overrides):
        path = Path(path)
        manifest = {
            "status": status,
            "task": task,
            "model_name": "test_model",
            "model_path": "/fake/model",
            "base_model": "meta-llama/Llama-3.1-8B",
            "run_id": "run_test",
            "backend": "vllm",
            "evaluation_mode": "generation_only",
            "predict_only": True,
            "judging_status": "pending_local",
            "server_side_accuracy_valid": False,
            "dtype": "bfloat16",
            "temperature": 0.0,
            "do_sample": False,
            "max_gen_toks": max_gen_toks,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": mnbt,
            "max_num_seqs": mns,
            "gpu_memory_utilization": gmu,
            "enable_prefix_caching": epc,
            "fallback_level": fallback_level,
            "tensor_parallel_size": 1,
            "successful_attempt": 1,
            "successful_attempt_elapsed_seconds": success_elapsed,
            "failed_attempt_elapsed_seconds": 0,
            "pipeline_elapsed_seconds": success_elapsed,
            "expected_sample_count": expected_count,
            "actual_sample_count": actual_count,
            "has_empty_output": False,
            "lm_eval_results_file": str(path.parent / "results.json"),
            "lm_eval_sample_file": str(path.parent / "samples.jsonl"),
            "completion_errors": [],
        }
        manifest.update(overrides)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return manifest

    return _make


@pytest.fixture
def make_mock_sample_jsonl():
    """生成 mock sample JSONL 文件（generation-only 格式）。"""
    def _make(path, n=2, token_counts=None, with_dup=False,
              empty_output=False, with_resps=True):
        rows = []
        token_counts = token_counts or [100, 200]
        for i in range(n):
            tc = token_counts[i % len(token_counts)]
            resps = [[f"answer {i}" * tc]]
            if not with_resps:
                resps = None
            if empty_output and i == 0:
                resps = [[""]]
            row = {
                "doc_id": i,
                "doc": {"problem": f"Problem {i}", "answer": str(i)},
                "arguments": [[f"Problem: Problem {i}\nAnswer:"]],
                "resps": resps,
                "filtered_resps": resps,
            }
            rows.append(row)
        if with_dup:
            rows.append({
                "doc_id": 0,
                "doc": {"problem": "Problem 0", "answer": "0"},
                "arguments": [["Problem: Problem 0\nAnswer:"]],
                "resps": [["duplicate"]],
                "filtered_resps": [["duplicate"]],
            })
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return rows

    return _make


@pytest.fixture
def make_mock_results_json():
    """生成 mock results JSON 文件。
    predict_only 模式下 results 可能只有 bypass 或空 dict。"""
    def _make(path, tasks=None, accs=None, predict_only=False):
        tasks = tasks or ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]
        results = {"results": {}}
        if predict_only:
            for t in tasks:
                results["results"][t] = {"bypass": 0.0}
        else:
            accs = accs or [0.5, 0.0, 0.5]
            for t, a in zip(tasks, accs):
                results["results"][t] = {"exact_match,none": a, "exact_match_stderr,none": 0.1}
        results["configs"] = {t: {} for t in tasks}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return results

    return _make


@pytest.fixture
def make_full_run_structure(tmp_path):
    """创建完整的 run 目录结构：active_run.json + runs/<run_id>/run_manifest.json + task manifests + sample JSONLs。"""
    def _make(model_dir_name="limo", tasks=None, n_samples=2,
              status="complete", predict_only=True,
              mnbt=8192, mns=32, gmu=0.90, epc=True,
              success_elapsed=1800, failed_elapsed=400,
              gpu_peak=42120, fallback_level=1,
              expected_counts=None):
        tasks = tasks or ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]
        model_dir = tmp_path / "results" / model_dir_name
        run_id = "run_test"
        run_dir = model_dir / "runs" / run_id

        # active_run.json
        model_dir.mkdir(parents=True, exist_ok=True)
        active = {"active_run_id": run_id, "status": status}
        with open(model_dir / "active_run.json", "w") as f:
            json.dump(active, f)

        # run dir
        run_dir.mkdir(parents=True, exist_ok=True)

        # task dirs
        expected = expected_counts or {t: n_samples for t in tasks}
        actual = {t: n_samples for t in tasks}
        sample_files = {}
        task_manifests = {}
        for task in tasks:
            task_dir = run_dir / "tasks" / task
            task_dir.mkdir(parents=True, exist_ok=True)
            # sample JSONL
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
            sample_files[task] = str(sample_path)

            # task manifest
            tm = {
                "status": status,
                "task": task,
                "model_name": model_dir_name,
                "model_path": f"/fake/{model_dir_name}",
                "evaluation_mode": "generation_only",
                "predict_only": predict_only,
                "judging_status": "pending_local",
                "max_gen_toks": 32768,
                "max_model_len": 40960,
                "max_num_batched_tokens": mnbt,
                "max_num_seqs": mns,
                "gpu_memory_utilization": gmu,
                "enable_prefix_caching": epc,
                "fallback_level": fallback_level,
                "successful_attempt_elapsed_seconds": success_elapsed,
                "expected_sample_count": expected.get(task, n_samples),
                "actual_sample_count": actual.get(task, n_samples),
                "lm_eval_sample_file": str(sample_path),
            }
            with open(task_dir / "task_manifest.json", "w") as f:
                json.dump(tm, f)
            task_manifests[task] = tm

        # run_manifest.json
        rm = {
            "status": status,
            "model_name": model_dir_name,
            "model_path": f"/fake/{model_dir_name}",
            "run_id": run_id,
            "tasks": tasks,
            "backend": "vllm",
            "evaluation_mode": "generation_only",
            "predict_only": predict_only,
            "judging_status": "pending_local",
            "max_gen_toks": 32768,
            "max_model_len": 40960,
            "max_num_batched_tokens": mnbt,
            "max_num_seqs": mns,
            "gpu_memory_utilization": gmu,
            "enable_prefix_caching": epc,
            "fallback_level": fallback_level,
            "gpu_peak_memory_mib": gpu_peak,
            "vllm_version": "0.6.6.post1",
            "lm_eval_version": "0.4.5",
            "dtype": "bfloat16",
            "pipeline_elapsed_seconds": success_elapsed + failed_elapsed,
            "successful_attempt_elapsed_seconds": success_elapsed,
            "failed_attempt_elapsed_seconds": failed_elapsed,
            "expected_sample_counts": expected,
            "actual_sample_counts": actual,
            "lm_eval_sample_files": sample_files,
        }
        with open(run_dir / "run_manifest.json", "w") as f:
            json.dump(rm, f)

        return {
            "model_dir": model_dir,
            "run_dir": run_dir,
            "run_manifest": rm,
            "task_manifests": task_manifests,
            "sample_files": sample_files,
        }

    return _make
