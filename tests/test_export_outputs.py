"""
test_export_outputs.py
测试 scripts/export_generation_outputs.py 的导出逻辑：
  * 行数正确
  * raw_output 完整保留
  * gold_answer 存在
  * hash 稳定
  * 原子写入（临时文件 + os.replace）
  * sample_id 唯一
  * 空输出检测
"""
import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

try:
    from export_generation_outputs import (
        export_single_task,
        load_task_manifest,
        _hash,
        _extract_resps,
        _extract_prompt,
        _extract_doc_id,
    )
except ImportError:
    pytest.skip("export_generation_outputs module not available", allow_module_level=True)


class TestHashStability:
    """hash 稳定性测试"""

    def test_same_text_same_hash(self):
        """相同文本应产生相同 hash"""
        h1 = _hash("hello world")
        h2 = _hash("hello world")
        assert h1 == h2

    def test_different_text_different_hash(self):
        """不同文本应产生不同 hash"""
        h1 = _hash("hello world")
        h2 = _hash("hello world!")
        assert h1 != h2

    def test_hash_length(self):
        """hash 长度为 16"""
        h = _hash("test")
        assert len(h) == 16


class TestExtractResps:
    """resps 提取测试"""

    def test_extract_from_resps(self):
        """从 resps 字段提取"""
        obj = {"resps": [["answer text"]]}
        raw, filtered = _extract_resps(obj)
        assert raw == "answer text"
        assert filtered == "answer text"

    def test_extract_from_filtered_resps(self):
        """从 filtered_resps 提取"""
        obj = {"resps": [["raw"]], "filtered_resps": [["filtered"]]}
        raw, filtered = _extract_resps(obj)
        assert raw == "raw"
        assert filtered == "filtered"

    def test_empty_resps(self):
        """空 resps"""
        obj = {"resps": [[""]]}
        raw, filtered = _extract_resps(obj)
        assert raw == ""
        assert filtered == ""

    def test_missing_resps(self):
        """缺少 resps"""
        obj = {}
        raw, filtered = _extract_resps(obj)
        assert raw == ""
        assert filtered == ""


class TestExtractPrompt:
    """prompt 提取测试"""

    def test_extract_from_arguments(self):
        """从 arguments 字段提取"""
        obj = {"arguments": [["prompt text", "stop1"]]}
        prompt = _extract_prompt(obj)
        assert prompt == "prompt text"

    def test_extract_from_prompt_field(self):
        """从 prompt 字段提取"""
        obj = {"prompt": "direct prompt"}
        prompt = _extract_prompt(obj)
        assert prompt == "direct prompt"

    def test_missing_prompt(self):
        """缺少 prompt"""
        obj = {}
        prompt = _extract_prompt(obj)
        assert prompt == ""


class TestExtractDocId:
    """doc_id 提取测试"""

    def test_extract_doc_id(self):
        obj = {"doc_id": 42}
        assert _extract_doc_id(obj, 0) == 42

    def test_extract_id(self):
        obj = {"id": "abc"}
        assert _extract_doc_id(obj, 0) == "abc"

    def test_fallback_to_line_num(self):
        obj = {}
        assert _extract_doc_id(obj, 5) == 5


class TestExportSingleTask:
    """export_single_task 完整测试"""

    def _create_mock_task_manifest(self, tmp_path, sample_file, task="local_math500_32k"):
        """创建 mock task manifest"""
        manifest = {
            "status": "complete",
            "task": task,
            "model_name": "LIMO-817",
            "model_path": "/fake/model",
            "base_model": "meta-llama/Llama-3.1-8B",
            "run_id": "run_test",
            "max_gen_toks": 32768,
            "max_model_len": 40960,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 32,
            "gpu_memory_utilization": 0.90,
            "enable_prefix_caching": True,
            "temperature": 0.0,
            "do_sample": False,
            "tensor_parallel_size": 1,
            "lm_eval_sample_file": str(sample_file),
        }
        return manifest

    def _create_sample_jsonl(self, path, n=3):
        """创建 mock sample JSONL"""
        with open(path, "w") as f:
            for i in range(n):
                f.write(json.dumps({
                    "doc_id": i,
                    "doc": {"problem": f"Problem {i}", "answer": str(i), "solution": f"Sol {i}"},
                    "arguments": [[f"Problem: Problem {i}\nAnswer:"]],
                    "resps": [[f"The answer is {i}"]],
                    "filtered_resps": [[f"The answer is {i}"]],
                }) + "\n")

    def test_export_correct_count(self, tmp_path):
        """导出行数正确"""
        sample_file = tmp_path / "samples.jsonl"
        self._create_sample_jsonl(sample_file, n=3)

        manifest = self._create_mock_task_manifest(tmp_path, sample_file)
        out_file = tmp_path / "exported.jsonl"

        result = export_single_task(manifest, "LIMO-817", "/fake/model", str(out_file))

        assert result["total_records"] == 3
        assert os.path.isfile(out_file)

        # 验证行数
        with open(out_file) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 3

    def test_raw_output_preserved(self, tmp_path):
        """raw_output 完整保留"""
        sample_file = tmp_path / "samples.jsonl"
        self._create_sample_jsonl(sample_file, n=2)

        manifest = self._create_mock_task_manifest(tmp_path, sample_file)
        out_file = tmp_path / "exported.jsonl"

        export_single_task(manifest, "LIMO-817", "/fake/model", str(out_file))

        with open(out_file) as f:
            for line in f:
                obj = json.loads(line)
                assert "raw_output" in obj
                assert obj["raw_output"]  # 非空
                assert "The answer is" in obj["raw_output"]

    def test_gold_answer_preserved(self, tmp_path):
        """gold_answer 被保留"""
        sample_file = tmp_path / "samples.jsonl"
        self._create_sample_jsonl(sample_file, n=2)

        manifest = self._create_mock_task_manifest(tmp_path, sample_file)
        out_file = tmp_path / "exported.jsonl"

        export_single_task(manifest, "LIMO-817", "/fake/model", str(out_file))

        with open(out_file) as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                assert "gold_answer" in obj
                assert obj["gold_answer"] == str(i)

    def test_sample_id_unique(self, tmp_path):
        """sample_id 唯一"""
        sample_file = tmp_path / "samples.jsonl"
        self._create_sample_jsonl(sample_file, n=5)

        manifest = self._create_mock_task_manifest(tmp_path, sample_file)
        out_file = tmp_path / "exported.jsonl"

        export_single_task(manifest, "LIMO-817", "/fake/model", str(out_file))

        ids = set()
        with open(out_file) as f:
            for line in f:
                obj = json.loads(line)
                sid = obj["sample_id"]
                assert sid not in ids, f"重复 sample_id: {sid}"
                ids.add(sid)
        assert len(ids) == 5

    def test_hash_stable(self, tmp_path):
        """相同输入产生相同 hash"""
        sample_file = tmp_path / "samples.jsonl"
        self._create_sample_jsonl(sample_file, n=2)

        manifest = self._create_mock_task_manifest(tmp_path, sample_file)
        out_file1 = tmp_path / "exported1.jsonl"
        out_file2 = tmp_path / "exported2.jsonl"

        export_single_task(manifest, "LIMO-817", "/fake/model", str(out_file1))
        export_single_task(manifest, "LIMO-817", "/fake/model", str(out_file2))

        # 读取两个文件的 hash
        hashes1 = []
        hashes2 = []
        with open(out_file1) as f:
            for line in f:
                obj = json.loads(line)
                hashes1.append(obj["prompt_hash"])
                hashes1.append(obj["output_hash"])
        with open(out_file2) as f:
            for line in f:
                obj = json.loads(line)
                hashes2.append(obj["prompt_hash"])
                hashes2.append(obj["output_hash"])

        assert hashes1 == hashes2

    def test_atomic_write(self, tmp_path):
        """原子写入：不应残留 .tmp 文件"""
        sample_file = tmp_path / "samples.jsonl"
        self._create_sample_jsonl(sample_file, n=2)

        manifest = self._create_mock_task_manifest(tmp_path, sample_file)
        out_file = tmp_path / "exported.jsonl"

        export_single_task(manifest, "LIMO-817", "/fake/model", str(out_file))

        # 不应残留 .tmp 文件
        assert not os.path.isfile(str(out_file) + ".tmp")
        assert os.path.isfile(out_file)

    def test_empty_output_detected(self, tmp_path):
        """空输出被检测"""
        sample_file = tmp_path / "samples.jsonl"
        with open(sample_file, "w") as f:
            f.write(json.dumps({
                "doc_id": 0,
                "doc": {"problem": "P0", "answer": "0"},
                "arguments": [["P0"]],
                "resps": [[""]],  # 空输出
            }) + "\n")
            f.write(json.dumps({
                "doc_id": 1,
                "doc": {"problem": "P1", "answer": "1"},
                "arguments": [["P1"]],
                "resps": [["answer 1"]],
            }) + "\n")

        manifest = self._create_mock_task_manifest(tmp_path, sample_file)
        out_file = tmp_path / "exported.jsonl"

        result = export_single_task(manifest, "LIMO-817", "/fake/model", str(out_file))

        assert result["empty_output_count"] == 1
        assert result["total_records"] == 2

    def test_duplicate_sample_id_raises(self, tmp_path):
        """重复 sample_id 应报错"""
        sample_file = tmp_path / "samples.jsonl"
        with open(sample_file, "w") as f:
            f.write(json.dumps({
                "doc_id": 0,
                "doc": {"problem": "P0", "answer": "0"},
                "arguments": [["P0"]],
                "resps": [["a"]],
            }) + "\n")
            f.write(json.dumps({
                "doc_id": 0,  # 重复 doc_id
                "doc": {"problem": "P0", "answer": "0"},
                "arguments": [["P0"]],
                "resps": [["b"]],
            }) + "\n")

        manifest = self._create_mock_task_manifest(tmp_path, sample_file)
        out_file = tmp_path / "exported.jsonl"

        with pytest.raises(ValueError, match="重复"):
            export_single_task(manifest, "LIMO-817", "/fake/model", str(out_file))

    def test_generation_config_in_output(self, tmp_path):
        """导出文件中应包含完整生成配置"""
        sample_file = tmp_path / "samples.jsonl"
        self._create_sample_jsonl(sample_file, n=1)

        manifest = self._create_mock_task_manifest(tmp_path, sample_file)
        out_file = tmp_path / "exported.jsonl"

        export_single_task(manifest, "LIMO-817", "/fake/model", str(out_file))

        with open(out_file) as f:
            obj = json.loads(f.readline())
            assert obj["max_gen_toks"] == 32768
            assert obj["max_model_len"] == 40960
            assert obj["max_num_batched_tokens"] == 8192
            assert obj["max_num_seqs"] == 32
            assert obj["gpu_memory_utilization"] == 0.90
            assert obj["enable_prefix_caching"] is True
            assert obj["temperature"] == 0.0
            assert obj["do_sample"] is False
            assert obj["tensor_parallel_size"] == 1

    def test_benchmark_name_correct(self, tmp_path):
        """不同 task 的 benchmark 名称正确"""
        for task, expected_bench in [
            ("local_math500_32k", "MATH500"),
            ("local_aime24_32k", "AIME24"),
            ("local_aime25_32k", "AIME25"),
        ]:
            sample_file = tmp_path / f"samples_{task}.jsonl"
            self._create_sample_jsonl(sample_file, n=1)

            manifest = self._create_mock_task_manifest(tmp_path, sample_file, task=task)
            out_file = tmp_path / f"exported_{task}.jsonl"

            export_single_task(manifest, "LIMO-817", "/fake/model", str(out_file))

            with open(out_file) as f:
                obj = json.loads(f.readline())
                assert obj["benchmark"] == expected_bench
                assert obj["task"] == task
