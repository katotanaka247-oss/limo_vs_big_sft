"""conftest.py — 为 tests/ 提供 sys.path 注入与通用 fixture。"""
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
    """提供一个临时项目目录，含 scripts/ 和 eval_tasks/ 子目录的符号链接。"""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "eval_tasks").mkdir()
    (tmp_path / "results").mkdir()
    return tmp_path


@pytest.fixture
def make_mock_manifest():
    """生成一个 mock manifest 并写入指定路径。"""
    import json

    def _make(path, status="complete", tasks=None, sample_counts=None,
              max_gen_toks=32768, max_model_len=40960, **overrides):
        tasks = tasks or ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]
        sample_files = {}
        for t in tasks:
            sample_files[t] = str(path.parent / f"samples_{t}.jsonl")
        manifest = {
            "status": status,
            "model_name": "test_model",
            "model_path": "/fake/model",
            "base_model": "meta-llama/Llama-3.1-8B",
            "run_id": "run_test",
            "tasks": tasks,
            "backend": "vllm",
            "dtype": "bfloat16",
            "temperature": 0.0,
            "do_sample": False,
            "max_gen_toks": max_gen_toks,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 32,
            "gpu_memory_utilization": 0.90,
            "enable_prefix_caching": True,
            "tensor_parallel_size": 1,
            "cuda_visible_devices": "0",
            "evaluation_protocol": "stock_zero_shot",
            "num_fewshot": 0,
            "apply_chat_template": False,
            "boxed_answer_instruction": False,
            "gpu_name": "NVIDIA L40",
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
            "elapsed_seconds": 3600,
            "successful_attempt": 1,
            "lm_eval_results_file": str(path.parent / "results.json"),
            "lm_eval_sample_files": sample_files,
            "completion_errors": [],
            "actual_sample_counts": sample_counts or {t: 2 for t in tasks},
        }
        manifest.update(overrides)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return manifest

    return _make


@pytest.fixture
def make_mock_sample_jsonl():
    """生成 mock sample JSONL 文件。"""
    import json

    def _make(path, n=2, token_counts=None, correct=None, with_dup=False):
        rows = []
        token_counts = token_counts or [100, 200]
        correct = correct or [True, False]
        for i in range(n):
            tc = token_counts[i % len(token_counts)]
            c = correct[i % len(correct)]
            rows.append({
                "doc_id": f"doc_{i}",
                "resps": [[f"answer {i}"]],
                "exact_match": 1.0 if c else 0.0,
                "response_tokens": tc,
            })
        if with_dup:
            # 添加一个重复 doc_id
            rows.append({
                "doc_id": "doc_0",
                "resps": [["duplicate"]],
                "exact_match": 0.0,
                "response_tokens": 50,
            })
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return rows

    return _make


@pytest.fixture
def make_mock_results_json():
    """生成 mock results JSON 文件。"""
    import json

    def _make(path, tasks=None, accs=None):
        tasks = tasks or ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]
        accs = accs or [0.5, 0.0, 0.5]
        results = {"results": {}}
        for t, a in zip(tasks, accs):
            results["results"][t] = {"exact_match,none": a, "exact_match_stderr,none": 0.1}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return results

    return _make
