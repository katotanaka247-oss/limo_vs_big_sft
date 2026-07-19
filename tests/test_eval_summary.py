"""
test_eval_summary.py
测试效率统计逻辑（generation-only 模式）：
  * _percentile 正确性（p 范围 0-100）
  * truncation 阈值判定
  * 两模型配置一致性检查（数值比较，非字符串比较）
  * tokens/s 使用 successful_attempt_elapsed_seconds
  * write_comparison 输出格式
  * 不报告 accuracy
"""
import json
from pathlib import Path

import pytest

from summarize_eval_efficiency import (
    _analyze_model,
    _configs_comparable,
    write_comparison,
    _percentile,
)


class TestPercentile:
    """P50/P90/P95 正确性（p 范围 0-100）"""

    def test_percentile_basic(self):
        vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        assert _percentile(vals, 50) == 55.0  # 插值
        assert _percentile(vals, 90) >= 90
        assert _percentile(vals, 95) >= 90

    def test_percentile_single(self):
        assert _percentile([42], 50) == 42

    def test_percentile_empty(self):
        assert _percentile([], 50) == 0

    def test_percentile_p90_exact(self):
        vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        result = _percentile(vals, 90)
        # k = (90/100) * 9 = 8.1, f=8, c=9
        # 90 + (100-90) * 0.1 = 91.0
        assert result == 91.0

    def test_percentile_p95_exact(self):
        vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        result = _percentile(vals, 95)
        # k = (95/100) * 9 = 8.55, f=8, c=9
        # 90 + 10 * 0.55 = 95.5
        assert abs(result - 95.5) < 0.01


class TestTruncationThreshold:
    """truncation 只根据 token 数判断（>= max_gen_toks - 8）"""

    def test_truncation_at_threshold(self, tmp_path, make_full_run_structure):
        structure = make_full_run_structure(model_dir_name="limo", n_samples=2)
        model_dir = structure["model_dir"]

        # 在 MATH500 的 sample 文件中，一个样本 token 数 >= 32760
        math500_sample = structure["sample_files"]["local_math500_32k"]
        with open(math500_sample, "w") as f:
            f.write(json.dumps({
                "doc_id": 0,
                "doc": {"problem": "Problem 0", "answer": "0"},
                "arguments": [["Problem: Problem 0\nAnswer:"]],
                "resps": [["short answer"]],
                "filtered_resps": [["short answer"]],
            }) + "\n")
            # 32760 个 word 的输出
            long_output = "word " * 32760
            f.write(json.dumps({
                "doc_id": 1,
                "doc": {"problem": "Problem 1", "answer": "1"},
                "arguments": [["Problem: Problem 1\nAnswer:"]],
                "resps": [[long_output]],
                "filtered_resps": [[long_output]],
            }) + "\n")

        stats = _analyze_model(str(model_dir), "LIMO-817")
        math500 = stats["task_stats"]["local_math500_32k"]
        assert math500["possibly_truncated_count"] == 1

    def test_no_truncation_below_threshold(self, tmp_path, make_full_run_structure):
        structure = make_full_run_structure(model_dir_name="limo", n_samples=2)
        model_dir = structure["model_dir"]

        stats = _analyze_model(str(model_dir), "LIMO-817")
        for task, ts in stats["task_stats"].items():
            assert ts["possibly_truncated_count"] == 0


class TestConfigComparison:
    """两模型配置一致性检查（数值比较，非字符串比较）"""

    def test_configs_match(self, tmp_path, make_full_run_structure):
        limo = make_full_run_structure(model_dir_name="limo", n_samples=2)
        openr1 = make_full_run_structure(model_dir_name="openr1", n_samples=2)

        limo_stats = _analyze_model(str(limo["model_dir"]), "LIMO-817")
        openr1_stats = _analyze_model(str(openr1["model_dir"]), "OpenR1-10K")

        assert _configs_comparable(limo_stats, openr1_stats) is True

    def test_configs_mismatch_max_num_seqs(self, tmp_path, make_full_run_structure):
        limo = make_full_run_structure(model_dir_name="limo", mns=32, n_samples=2)
        openr1 = make_full_run_structure(model_dir_name="openr1", mns=8, n_samples=2)

        limo_stats = _analyze_model(str(limo["model_dir"]), "LIMO-817")
        openr1_stats = _analyze_model(str(openr1["model_dir"]), "OpenR1-10K")

        assert _configs_comparable(limo_stats, openr1_stats) is False

    def test_configs_float_equal(self, tmp_path, make_full_run_structure):
        """0.90 和 0.9 应被视为相等（浮点数值比较）"""
        limo = make_full_run_structure(model_dir_name="limo", gmu=0.90, n_samples=2)
        openr1 = make_full_run_structure(model_dir_name="openr1", gmu=0.9, n_samples=2)

        limo_stats = _analyze_model(str(limo["model_dir"]), "LIMO-817")
        openr1_stats = _analyze_model(str(openr1["model_dir"]), "OpenR1-10K")

        assert _configs_comparable(limo_stats, openr1_stats) is True

    def test_configs_float_not_equal(self, tmp_path, make_full_run_structure):
        """0.90 和 0.88 应被视为不等"""
        limo = make_full_run_structure(model_dir_name="limo", gmu=0.90, n_samples=2)
        openr1 = make_full_run_structure(model_dir_name="openr1", gmu=0.88, n_samples=2)

        limo_stats = _analyze_model(str(limo["model_dir"]), "LIMO-817")
        openr1_stats = _analyze_model(str(openr1["model_dir"]), "OpenR1-10K")

        assert _configs_comparable(limo_stats, openr1_stats) is False

    def test_write_comparison_config_match(self, tmp_path, make_full_run_structure):
        limo = make_full_run_structure(model_dir_name="limo", n_samples=2)
        openr1 = make_full_run_structure(model_dir_name="openr1", n_samples=2)

        limo_stats = _analyze_model(str(limo["model_dir"]), "LIMO-817")
        openr1_stats = _analyze_model(str(openr1["model_dir"]), "OpenR1-10K")

        out_json = str(tmp_path / "comparison.json")
        out_csv = str(tmp_path / "comparison.csv")
        out_md = str(tmp_path / "comparison.md")

        write_comparison(limo_stats, openr1_stats, out_json, out_csv, out_md)

        with open(out_json, encoding="utf-8") as f:
            comp = json.load(f)
        assert comp["config_comparable"] is True
        assert comp["throughput_comparable"] is True
        assert comp["generation_mode"] == "generation_only"
        assert comp["accuracy_reported"] is False

    def test_write_comparison_config_mismatch(self, tmp_path, make_full_run_structure):
        limo = make_full_run_structure(model_dir_name="limo", mns=32, n_samples=2)
        openr1 = make_full_run_structure(model_dir_name="openr1", mns=8, n_samples=2)

        limo_stats = _analyze_model(str(limo["model_dir"]), "LIMO-817")
        openr1_stats = _analyze_model(str(openr1["model_dir"]), "OpenR1-10K")

        out_json = str(tmp_path / "comparison.json")
        out_csv = str(tmp_path / "comparison.csv")
        out_md = str(tmp_path / "comparison.md")

        write_comparison(limo_stats, openr1_stats, out_json, out_csv, out_md)

        with open(out_json, encoding="utf-8") as f:
            comp = json.load(f)
        assert comp["config_comparable"] is False
        assert comp["throughput_comparable"] is False


class TestTimingStats:
    """tokens/s 必须使用 successful_attempt_elapsed_seconds"""

    def test_tokens_per_s_uses_success_time(self, tmp_path, make_full_run_structure):
        """验证 tokens_per_s 使用成功 attempt 耗时，而非 pipeline 耗时"""
        structure = make_full_run_structure(
            model_dir_name="limo", n_samples=2,
            success_elapsed=300, failed_elapsed=100,
        )
        stats = _analyze_model(str(structure["model_dir"]), "LIMO-817")
        assert stats["successful_attempt_wall_time"] == 300
        assert stats["pipeline_elapsed_seconds"] == 400
        assert stats["successful_attempt_wall_time"] != stats["pipeline_elapsed_seconds"]
        if stats["total_output_tokens"] > 0:
            expected_tps = round(stats["total_output_tokens"] / 300, 1)
            assert stats["tokens_per_s"] == expected_tps

    def test_manifest_timing_fields_present(self, tmp_path, make_full_run_structure):
        structure = make_full_run_structure(
            model_dir_name="limo", n_samples=2,
            success_elapsed=300, failed_elapsed=100,
        )
        stats = _analyze_model(str(structure["model_dir"]), "LIMO-817")
        assert "successful_attempt_wall_time" in stats
        assert "pipeline_elapsed_seconds" in stats
        assert "tokens_per_s" in stats

    def test_zero_success_elapsed_no_division_error(self, tmp_path, make_full_run_structure):
        """success_elapsed=0 时不应除零"""
        structure = make_full_run_structure(
            model_dir_name="limo", n_samples=2,
            success_elapsed=0, failed_elapsed=0,
        )
        stats = _analyze_model(str(structure["model_dir"]), "LIMO-817")
        assert stats["tokens_per_s"] == 0


class TestGenerationOnlyStats:
    """generation-only 模式统计字段"""

    def test_no_accuracy_in_stats(self, tmp_path, make_full_run_structure):
        """stats 中不应包含 accuracy 字段"""
        structure = make_full_run_structure(model_dir_name="limo", n_samples=2)
        stats = _analyze_model(str(structure["model_dir"]), "LIMO-817")
        assert "accuracy" not in stats
        assert "exact_match" not in stats
        assert "acc" not in stats

    def test_generation_complete_flag(self, tmp_path, make_full_run_structure):
        structure = make_full_run_structure(model_dir_name="limo", n_samples=2,
                                            status="complete")
        stats = _analyze_model(str(structure["model_dir"]), "LIMO-817")
        assert stats["generation_complete"] is True

    def test_generation_incomplete_when_status_incomplete(self, tmp_path, make_full_run_structure):
        structure = make_full_run_structure(model_dir_name="limo", n_samples=2,
                                            status="incomplete")
        stats = _analyze_model(str(structure["model_dir"]), "LIMO-817")
        assert stats["generation_complete"] is False

    def test_generation_complete_false_with_empty_outputs(self, tmp_path, make_full_run_structure):
        """空输出应导致 generation_complete=False"""
        structure = make_full_run_structure(model_dir_name="limo", n_samples=2)
        model_dir = structure["model_dir"]

        # 在 MATH500 的 sample 文件中写入空输出
        math500_sample = structure["sample_files"]["local_math500_32k"]
        with open(math500_sample, "w") as f:
            for i in range(2):
                f.write(json.dumps({
                    "doc_id": i,
                    "doc": {"problem": f"Problem {i}", "answer": str(i)},
                    "arguments": [["Problem:"]],
                    "resps": [[""]],
                    "filtered_resps": [[""]],
                }) + "\n")

        stats = _analyze_model(str(model_dir), "LIMO-817")
        assert stats["generation_complete"] is False
        assert stats["empty_output_count"] > 0

    def test_gpu_peak_memory_in_stats(self, tmp_path, make_full_run_structure):
        structure = make_full_run_structure(model_dir_name="limo", n_samples=2,
                                            gpu_peak=42120)
        stats = _analyze_model(str(structure["model_dir"]), "LIMO-817")
        assert stats["gpu_peak_memory_mib"] == 42120

    def test_fallback_level_in_stats(self, tmp_path, make_full_run_structure):
        structure = make_full_run_structure(model_dir_name="limo", n_samples=2,
                                            fallback_level=2)
        stats = _analyze_model(str(structure["model_dir"]), "LIMO-817")
        assert stats["fallback_level"] == 2
