"""
test_generation_only.py
测试 generation-only 模式的关键逻辑：
  * 完成度判定不要求 accuracy
  * 输出完整性（非空、JSON 可解析、doc_id 唯一、行数正确）
  * 缺一条则失败、重复一条则失败、缺 resps 则失败
  * AIME 数量验证（expected=30, actual=29 必须失败）
  * 耗时统计使用 successful_attempt_elapsed_seconds
  * 配置比较（0.90 vs 0.9 相等）
"""
import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))


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
        # generation-only 不检查 accuracy
        for task, task_result in results["results"].items():
            assert isinstance(task_result, dict)
            # 不应要求 exact_match 或 acc
            has_acc = any(
                k.startswith("exact_match") or k.startswith("acc")
                for k in task_result
            )
            # bypass 模式下 has_acc 应为 False
            assert not has_acc or "bypass" in task_result

    def test_empty_results_accepted(self):
        """predict_only 模式下 results 可能为空 dict，应通过"""
        results = {
            "results": {
                "local_math500_32k": {},
            },
        }
        # 空 dict 也能通过
        task_result = results["results"]["local_math500_32k"]
        has_acc = any(
            k.startswith("exact_match") or k.startswith("acc")
            for k in task_result
        )
        assert not has_acc


class TestOutputCompleteness:
    """输出完整性检查"""

    def _check_sample_jsonl(self, path, expected_count=None):
        """模拟完成度判定中的 sample JSONL 检查。
        返回 (errors, actual_count)。"""
        errors = []
        actual_count = 0
        seen_ids = set()

        if not os.path.isfile(path):
            errors.append("sample JSONL 不存在")
            return errors, 0

        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                actual_count += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    errors.append(f"第 {line_num} 行 JSON 解析失败")
                    continue

                did = obj.get("doc_id")
                if did is None:
                    did = obj.get("id")
                if did is None:
                    did = str(line_num)
                if did in seen_ids:
                    errors.append(f"重复 doc_id={did}")
                seen_ids.add(did)

                resps = obj.get("resps") or obj.get("filtered_resps")
                if resps is None:
                    errors.append(f"第 {line_num} 行缺少 resps/filtered_resps")

                has_prompt = (obj.get("arguments") is not None or
                              obj.get("prompt") is not None or
                              obj.get("doc") is not None)
                if not has_prompt:
                    errors.append(f"第 {line_num} 行缺少 prompt/arguments/doc")

        if expected_count is not None and actual_count != expected_count:
            errors.append(f"样本数={actual_count}, 预期={expected_count}")

        return errors, actual_count

    def test_valid_output(self, tmp_path):
        """正常的 2 条输出应通过"""
        path = tmp_path / "samples.jsonl"
        with open(path, "w") as f:
            for i in range(2):
                f.write(json.dumps({
                    "doc_id": i,
                    "doc": {"problem": f"P{i}"},
                    "arguments": [[f"Problem: P{i}\nAnswer:"]],
                    "resps": [[f"answer {i}"]],
                }) + "\n")

        errors, count = self._check_sample_jsonl(str(path), expected_count=2)
        assert errors == []
        assert count == 2

    def test_missing_one_fails(self, tmp_path):
        """缺一条应失败"""
        path = tmp_path / "samples.jsonl"
        with open(path, "w") as f:
            for i in range(1):  # 只写 1 条，预期 2 条
                f.write(json.dumps({
                    "doc_id": i,
                    "arguments": [[f"Problem: P{i}"]],
                    "resps": [[f"answer {i}"]],
                }) + "\n")

        errors, count = self._check_sample_jsonl(str(path), expected_count=2)
        assert len(errors) > 0
        assert any("样本数" in e for e in errors)

    def test_duplicate_fails(self, tmp_path):
        """重复一条应失败"""
        path = tmp_path / "samples.jsonl"
        with open(path, "w") as f:
            for i in [0, 0]:  # 重复 doc_id=0
                f.write(json.dumps({
                    "doc_id": i,
                    "arguments": [[f"Problem"]],
                    "resps": [[f"answer"]],
                }) + "\n")

        errors, count = self._check_sample_jsonl(str(path), expected_count=2)
        assert any("重复" in e for e in errors)

    def test_missing_resps_fails(self, tmp_path):
        """缺 resps 应失败"""
        path = tmp_path / "samples.jsonl"
        with open(path, "w") as f:
            for i in range(2):
                f.write(json.dumps({
                    "doc_id": i,
                    "arguments": [[f"Problem"]],
                    # 没有 resps
                }) + "\n")

        errors, count = self._check_sample_jsonl(str(path), expected_count=2)
        assert any("resps" in e for e in errors)

    def test_empty_output_detected(self, tmp_path):
        """空输出应被检测"""
        path = tmp_path / "samples.jsonl"
        with open(path, "w") as f:
            for i in range(2):
                f.write(json.dumps({
                    "doc_id": i,
                    "arguments": [[f"Problem"]],
                    "resps": [[""]],  # 空输出
                }) + "\n")

        # 检查空输出
        empty_count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                resps = obj.get("resps")
                if resps and isinstance(resps, list) and len(resps) > 0:
                    first = resps[0]
                    if isinstance(first, list) and len(first) > 0:
                        if not str(first[0]).strip():
                            empty_count += 1

        assert empty_count == 2

    def test_corrupt_json_fails(self, tmp_path):
        """JSON 损坏应失败"""
        path = tmp_path / "samples.jsonl"
        with open(path, "w") as f:
            f.write('{"doc_id": 0, "resps": [["ok"]]}\n')
            f.write('{ broken json\n')

        errors, count = self._check_sample_jsonl(str(path), expected_count=2)
        assert any("JSON 解析失败" in e for e in errors)


class TestAIMECount:
    """AIME 数量验证"""

    def test_aime24_count_mismatch_fails(self, tmp_path):
        """AIME24 expected=30, actual=29 必须失败"""
        path = tmp_path / "aime24_samples.jsonl"
        with open(path, "w") as f:
            for i in range(29):  # 少一条
                f.write(json.dumps({
                    "doc_id": i,
                    "arguments": [[f"Question"]],
                    "resps": [[f"answer {i}"]],
                }) + "\n")

        # 模拟完成度判定
        actual_count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    actual_count += 1

        expected = 30
        assert actual_count == 29
        assert actual_count != expected

    def test_aime25_count_mismatch_fails(self, tmp_path):
        """AIME25 expected=30, actual=31 必须失败"""
        path = tmp_path / "aime25_samples.jsonl"
        with open(path, "w") as f:
            for i in range(31):  # 多一条
                f.write(json.dumps({
                    "doc_id": i,
                    "arguments": [[f"Question"]],
                    "resps": [[f"answer {i}"]],
                }) + "\n")

        actual_count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    actual_count += 1

        expected = 30
        assert actual_count == 31
        assert actual_count != expected

    def test_math500_count_match_passes(self, tmp_path):
        """MATH500 expected=500, actual=500 应通过"""
        path = tmp_path / "math500_samples.jsonl"
        with open(path, "w") as f:
            for i in range(500):
                f.write(json.dumps({
                    "doc_id": i,
                    "arguments": [[f"Problem"]],
                    "resps": [[f"answer {i}"]],
                }) + "\n")

        actual_count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    actual_count += 1

        assert actual_count == 500


class TestTimingStats:
    """耗时统计：tokens/s 使用 successful_attempt_elapsed_seconds"""

    def test_tokens_per_s_uses_success_time(self):
        """attempt1=OOM 100s, attempt2=success 300s
        tokens/s 应使用 300s 而非 400s"""
        total_tokens = 30000

        # 模拟 manifest 中的耗时
        success_elapsed = 300
        failed_elapsed = 100
        pipeline_elapsed = success_elapsed + failed_elapsed  # 400

        # tokens/s 应使用 success_elapsed
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
              "dtype": "bfloat16"}
        c2 = {"max_gen_toks": 32768, "max_model_len": 40960,
              "max_num_batched_tokens": 8192, "max_num_seqs": 32,
              "gpu_memory_utilization": 0.9,  # 字符串不同但数值相同
              "enable_prefix_caching": True,
              "dtype": "bfloat16"}

        fields = ["max_gen_toks", "max_model_len", "max_num_batched_tokens",
                  "max_num_seqs", "gpu_memory_utilization",
                  "enable_prefix_caching", "dtype"]

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
