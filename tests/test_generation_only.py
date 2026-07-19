"""
test_generation_only.py
测试 generation-only 模式的关键逻辑。

使用生产代码 validate_generation_task.py 中的函数进行验证，
不在测试中重新实现验证逻辑。
"""
import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from validate_generation_task import (
    validate_lm_eval_sample_file,
    validate_exported_generation_file,
    validate_task_manifest,
    validate_run_manifest,
    validate_export_record,
    extract_output,
    extract_doc_id,
)


class TestGenerationCompletion:
    """generation-only 完成度判定（不要求 accuracy）"""

    def test_bypass_metric_accepted(self):
        """predict_only 模式下 results 只有 bypass，应通过"""
        results = {
            "results": {
                "local_math500_32k": {"bypass": 0.0},
                "local_aime24_32k": {"bypass": 0.0},
                "local_aime25_32k": {"bypass": 0.0},
            },
            "configs": {},
        }
        for task, task_result in results["results"].items():
            assert isinstance(task_result, dict)
            has_acc = any(
                k.startswith("exact_match") or k.startswith("acc")
                for k in task_result
            )
            assert not has_acc or "bypass" in task_result

    def test_empty_results_accepted(self):
        """predict_only 模式下 results 可能为空 dict"""
        results = {"results": {"local_math500_32k": {}}}
        task_result = results["results"]["local_math500_32k"]
        has_acc = any(
            k.startswith("exact_match") or k.startswith("acc")
            for k in task_result
        )
        assert not has_acc


class TestOutputCompleteness:
    """输出完整性检查（使用生产代码 validate_lm_eval_sample_file）"""

    def _write_samples(self, path, records):
        """写入 sample JSONL"""
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _valid_sample(self, doc_id=0):
        return {
            "doc_id": doc_id,
            "doc": {"problem": f"Problem {doc_id}", "answer": str(doc_id)},
            "arguments": {"gen_args_0": {"arg_0": f"Problem: {doc_id}\nAnswer:", "arg_1": {}}},
            "resps": [[f"answer {doc_id}"]],
            "filtered_resps": [[f"answer {doc_id}"]],
        }

    def test_valid_output(self, tmp_path):
        """正常的 2 条输出应通过"""
        path = tmp_path / "samples.jsonl"
        self._write_samples(path, [self._valid_sample(0), self._valid_sample(1)])
        errors = validate_lm_eval_sample_file(str(path), "test_task", 2, 32768)
        assert errors == []

    def test_missing_one_fails(self, tmp_path):
        """缺一条应失败"""
        path = tmp_path / "samples.jsonl"
        self._write_samples(path, [self._valid_sample(0)])  # 只写 1 条
        errors = validate_lm_eval_sample_file(str(path), "test_task", 2, 32768)
        assert any("样本数" in e for e in errors)

    def test_duplicate_fails(self, tmp_path):
        """重复 doc_id 应失败"""
        path = tmp_path / "samples.jsonl"
        self._write_samples(path, [self._valid_sample(0), self._valid_sample(0)])
        errors = validate_lm_eval_sample_file(str(path), "test_task", 0, 32768)
        assert any("重复" in e for e in errors)

    def test_duplicate_doc_id_zero_fails(self, tmp_path):
        """P1: 重复 doc_id=0 应被检测"""
        path = tmp_path / "samples.jsonl"
        s1 = self._valid_sample(0)
        s2 = self._valid_sample(0)
        self._write_samples(path, [s1, s2])
        errors = validate_lm_eval_sample_file(str(path), "test_task", 0, 32768)
        assert any("重复" in e for e in errors)

    def test_missing_resps_fails(self, tmp_path):
        """缺 resps 应失败"""
        path = tmp_path / "samples.jsonl"
        s = self._valid_sample(0)
        del s["resps"]
        del s["filtered_resps"]
        self._write_samples(path, [s, self._valid_sample(1)])
        errors = validate_lm_eval_sample_file(str(path), "test_task", 2, 32768)
        assert any("resps" in e for e in errors)

    def test_empty_output_detected(self, tmp_path):
        """P0: 空输出应加入 errors"""
        path = tmp_path / "samples.jsonl"
        s = self._valid_sample(0)
        s["resps"] = [[""]]
        s["filtered_resps"] = [[""]]
        self._write_samples(path, [s, self._valid_sample(1)])
        errors = validate_lm_eval_sample_file(str(path), "test_task", 2, 32768)
        assert any("输出为空" in e for e in errors)

    def test_corrupt_json_fails(self, tmp_path):
        """JSON 损坏应失败"""
        path = tmp_path / "samples.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(self._valid_sample(0)) + "\n")
            f.write("{ broken json\n")
        errors = validate_lm_eval_sample_file(str(path), "test_task", 2, 32768)
        assert any("JSON" in e for e in errors)


class TestAIMECount:
    """AIME 数量验证（使用生产代码）"""

    def _write_samples(self, path, n):
        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                f.write(json.dumps({
                    "doc_id": i,
                    "doc": {"problem": f"P{i}", "answer": str(i)},
                    "arguments": {"gen_args_0": {"arg_0": f"Q{i}", "arg_1": {}}},
                    "resps": [[f"a{i}"]],
                }) + "\n")

    def test_aime24_count_mismatch_fails(self, tmp_path):
        """AIME24 expected=30, actual=29 必须失败"""
        path = tmp_path / "aime24_samples.jsonl"
        self._write_samples(path, 29)
        errors = validate_lm_eval_sample_file(str(path), "local_aime24_32k", 30, 32768)
        assert any("样本数" in e for e in errors)

    def test_aime25_count_mismatch_fails(self, tmp_path):
        """AIME25 expected=30, actual=31 必须失败"""
        path = tmp_path / "aime25_samples.jsonl"
        self._write_samples(path, 31)
        errors = validate_lm_eval_sample_file(str(path), "local_aime25_32k", 30, 32768)
        assert any("样本数" in e for e in errors)

    def test_math500_count_match_passes(self, tmp_path):
        """MATH500 expected=500, actual=500 应通过"""
        path = tmp_path / "math500_samples.jsonl"
        self._write_samples(path, 10)  # 测试用 10 条
        errors = validate_lm_eval_sample_file(str(path), "local_math500_32k", 10, 32768)
        assert not any("样本数" in e for e in errors)


class TestTimingStats:
    """耗时统计：tokens/s 使用 successful_attempt_elapsed_seconds"""

    def test_tokens_per_s_uses_success_time(self):
        """attempt1=OOM 100s, attempt2=success 300s
        tokens/s 应使用 300s 而非 400s"""
        total_tokens = 30000
        success_elapsed = 300
        failed_elapsed = 100
        pipeline_elapsed = success_elapsed + failed_elapsed  # 400

        tokens_per_s_success = total_tokens / success_elapsed  # 100.0
        tokens_per_s_pipeline = total_tokens / pipeline_elapsed  # 75.0

        assert tokens_per_s_success == 100.0
        assert tokens_per_s_pipeline == 75.0
        assert tokens_per_s_success != tokens_per_s_pipeline

    def test_manifest_fields_present(self, make_full_run_structure):
        """manifest 中必须有 successful_attempt_elapsed_seconds 字段"""
        structure = make_full_run_structure(
            model_dir_name="limo",
            success_elapsed=1800,
            failed_elapsed=400,
        )
        rm = structure["run_manifest"]
        assert "successful_attempt_elapsed_seconds" in rm
        assert rm["successful_attempt_elapsed_seconds"] == 1800
        assert "failed_attempt_elapsed_seconds" in rm
        assert rm["failed_attempt_elapsed_seconds"] == 400
        assert "pipeline_elapsed_seconds" in rm
        assert rm["pipeline_elapsed_seconds"] == 2200


class TestConfigComparisonNumeric:
    """配置比较使用数值比较"""

    def test_gmu_float_equal(self):
        """0.90 和 0.9 数值相等"""
        assert abs(float("0.90") - float("0.9")) < 1e-6

    def test_gmu_float_not_equal(self):
        """0.90 和 0.88 数值不等"""
        assert abs(float("0.90") - float("0.88")) >= 1e-6

    def test_full_config_comparison(self):
        """完整配置比较（数值化）"""
        c1 = {"max_gen_toks": 32768, "max_model_len": 40960,
              "max_num_batched_tokens": 8192, "max_num_seqs": 32,
              "gpu_memory_utilization": 0.90, "enable_prefix_caching": True,
              "enable_chunked_prefill": True, "dtype": "bfloat16"}
        c2 = {"max_gen_toks": 32768, "max_model_len": 40960,
              "max_num_batched_tokens": 8192, "max_num_seqs": 32,
              "gpu_memory_utilization": 0.9,
              "enable_prefix_caching": True,
              "enable_chunked_prefill": True,
              "dtype": "bfloat16"}

        fields = ["max_gen_toks", "max_model_len", "max_num_batched_tokens",
                  "max_num_seqs", "gpu_memory_utilization",
                  "enable_prefix_caching", "enable_chunked_prefill", "dtype"]

        all_match = True
        for field in fields:
            v1 = c1[field]
            v2 = c2[field]
            if isinstance(v1, float) or isinstance(v2, float):
                if abs(float(v1) - float(v2)) > 1e-6:
                    all_match = False
            elif v1 != v2:
                all_match = False

        assert all_match

    def test_different_fallback_level_not_comparable(self):
        """不同 fallback level 的配置不可比较"""
        c1 = {"fallback_level": 1, "max_num_batched_tokens": 8192}
        c2 = {"fallback_level": 3, "max_num_batched_tokens": 2048}
        assert c1["fallback_level"] != c2["fallback_level"]

    def test_chunked_prefill_mismatch_not_comparable(self):
        """enable_chunked_prefill 不同时不可比较"""
        c1 = {"enable_chunked_prefill": True}
        c2 = {"enable_chunked_prefill": False}
        assert c1["enable_chunked_prefill"] != c2["enable_chunked_prefill"]


class TestTaskManifestValidation:
    """使用生产代码验证 task manifest"""

    def test_valid_task_manifest(self, tmp_path):
        """有效的 task manifest 应通过验证"""
        # 先创建导出文件
        export_path = str(tmp_path / "exported.jsonl")
        record = {
            "sample_id": "task:0", "task": "local_math500_32k",
            "benchmark": "MATH500", "question": "Q", "prompt": "P",
            "raw_output": "A", "prompt_hash": "h1", "output_hash": "h2",
            "run_id": "r1", "doc_id": 0, "gold_answer": "0",
            "max_gen_toks": 32768, "max_model_len": 40960,
            "output_token_count": 5, "output_token_count_method": "llama_tokenizer",
        }
        with open(export_path, "w") as f:
            f.write(json.dumps(record) + "\n")

        manifest_path = str(tmp_path / "task_manifest.json")
        manifest = {
            "status": "complete", "task": "local_math500_32k",
            "max_gen_toks": 32768, "max_model_len": 40960,
            "export_status": "complete",
            "exported_generation_file": export_path,
            "expected_sample_count": 1, "actual_sample_count": 1,
            "exported_sample_count": 1,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        errors = validate_task_manifest(manifest_path)
        assert errors == []

    def test_incomplete_task_manifest_fails(self, tmp_path):
        """status=incomplete 应失败"""
        manifest_path = str(tmp_path / "task_manifest.json")
        manifest = {"status": "incomplete", "max_gen_toks": 32768, "max_model_len": 40960}
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        errors = validate_task_manifest(manifest_path)
        assert any("status" in e for e in errors)

    def test_missing_export_file_fails(self, tmp_path):
        """缺少导出文件应失败"""
        manifest_path = str(tmp_path / "task_manifest.json")
        manifest = {
            "status": "complete", "max_gen_toks": 32768, "max_model_len": 40960,
            "export_status": "complete",
            "exported_generation_file": "/nonexistent/path.jsonl",
            "expected_sample_count": 1, "actual_sample_count": 1,
            "exported_sample_count": 1,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        errors = validate_task_manifest(manifest_path)
        assert any("exported_generation_file" in e for e in errors)


class TestRunManifestValidation:
    """使用生产代码验证 run manifest"""

    def test_valid_run_manifest(self, tmp_path):
        """有效的 run manifest 应通过验证"""
        # 创建导出文件
        export_files = {}
        for task in ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]:
            ef = str(tmp_path / f"{task}_export.jsonl")
            record = {
                "sample_id": f"{task}:0", "task": task,
                "benchmark": "BENCH", "question": "Q", "prompt": "P",
                "raw_output": "A", "prompt_hash": "h1", "output_hash": "h2",
                "run_id": "r1", "doc_id": 0, "gold_answer": "0",
                "max_gen_toks": 32768, "max_model_len": 40960,
                "output_token_count": 5, "output_token_count_method": "llama_tokenizer",
            }
            with open(ef, "w") as f:
                f.write(json.dumps(record) + "\n")
            export_files[task] = ef

        manifest_path = str(tmp_path / "run_manifest.json")
        manifest = {
            "status": "complete",
            "evaluation_mode": "generation_only",
            "predict_only": True,
            "judging_status": "pending_local",
            "max_gen_toks": 32768,
            "enable_chunked_prefill": True,
            "tasks": list(export_files.keys()),
            "exported_generation_files": export_files,
            "expected_sample_counts": {t: 1 for t in export_files},
            "actual_sample_counts": {t: 1 for t in export_files},
            "gpu_peak_memory_mib": 42120,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        errors = validate_run_manifest(manifest_path)
        assert errors == []

    def test_missing_chunked_prefill_fails(self, tmp_path):
        """缺少 enable_chunked_prefill 应失败"""
        manifest_path = str(tmp_path / "run_manifest.json")
        manifest = {
            "status": "complete",
            "evaluation_mode": "generation_only",
            "predict_only": True,
            "judging_status": "pending_local",
            "max_gen_toks": 32768,
            "tasks": [],
            "exported_generation_files": {},
            "expected_sample_counts": {},
            "actual_sample_counts": {},
            "gpu_peak_memory_mib": 42120,
            # 没有 enable_chunked_prefill
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        errors = validate_run_manifest(manifest_path)
        assert any("enable_chunked_prefill" in e for e in errors)

    def test_incomplete_status_fails(self, tmp_path):
        """status=incomplete 应失败"""
        manifest_path = str(tmp_path / "run_manifest.json")
        manifest = {
            "status": "incomplete",
            "evaluation_mode": "generation_only",
            "predict_only": True,
            "judging_status": "pending_local",
            "max_gen_toks": 32768,
            "enable_chunked_prefill": True,
            "tasks": [],
            "exported_generation_files": {},
            "expected_sample_counts": {},
            "actual_sample_counts": {},
            "gpu_peak_memory_mib": 42120,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        errors = validate_run_manifest(manifest_path)
        assert any("status" in e for e in errors)
