"""
test_export_outputs.py
测试统一 JSONL 导出逻辑。

关键测试:
  - extract_prompt 兼容 lm-eval 0.4.5 dict 格式
  - extract_prompt 兼容 raw list 格式
  - 空输出导致导出失败
  - 空 prompt 导致导出失败
  - doc_id=0 重复检测
  - 严格字段验证
  - token count method 必须为 llama_tokenizer
  - 原子写入
  - hash 稳定性
"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 确保能 import 生产代码
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from validate_generation_task import (
    extract_output,
    extract_prompt as _extract_prompt,
    extract_doc_id,
    validate_export_record,
    validate_lm_eval_sample_file,
    validate_exported_generation_file,
)
from export_generation_outputs import (
    _hash,
    export_single_task,
    _count_tokens,
)


# ---------- extract_prompt 测试 ----------

class TestExtractPromptLmEval045:
    """P0: extract_prompt 兼容 lm-eval 0.4.5 dict 格式"""

    def test_extract_prompt_from_lm_eval_045_dict(self):
        """lm-eval 0.4.5 保存后的 dict 格式: gen_args_0.arg_0"""
        sample = {
            "arguments": {
                "gen_args_0": {
                    "arg_0": "Problem: 1+1\nAnswer:",
                    "arg_1": {
                        "max_gen_toks": 32768
                    }
                }
            }
        }
        assert _extract_prompt(sample) == "Problem: 1+1\nAnswer:"

    def test_extract_prompt_from_lm_eval_045_dict_with_multiple_keys(self):
        """多 key 时按排序取第一个有 arg_0 的"""
        sample = {
            "arguments": {
                "gen_args_1": {"arg_0": "second", "arg_1": {}},
                "gen_args_0": {"arg_0": "first", "arg_1": {}},
            }
        }
        # gen_args_0 应该优先
        assert _extract_prompt(sample) == "first"

    def test_extract_prompt_from_lm_eval_045_dict_arg_0_none(self):
        """arg_0 为 None 时应回退"""
        sample = {
            "arguments": {
                "gen_args_0": {"arg_0": None, "arg_1": {}},
                "gen_args_1": {"arg_0": "fallback", "arg_1": {}},
            }
        }
        assert _extract_prompt(sample) == "fallback"


class TestExtractPromptRawList:
    """P0: extract_prompt 兼容 raw list 格式"""

    def test_extract_prompt_from_raw_list(self):
        """lm-eval 未序列化前: [["prompt", {...}]]"""
        sample = {
            "arguments": [
                ["Problem: 1+1\nAnswer:", {"max_gen_toks": 32768}]
            ]
        }
        assert _extract_prompt(sample) == "Problem: 1+1\nAnswer:"

    def test_extract_prompt_from_raw_tuple(self):
        """tuple 格式"""
        sample = {
            "arguments": [
                ("Problem: 1+1\nAnswer:", {"max_gen_toks": 32768})
            ]
        }
        assert _extract_prompt(sample) == "Problem: 1+1\nAnswer:"

    def test_extract_prompt_from_string_list(self):
        """纯字符串列表"""
        sample = {"arguments": ["just a prompt"]}
        assert _extract_prompt(sample) == "just a prompt"

    def test_extract_prompt_from_empty_list(self):
        """空列表"""
        sample = {"arguments": []}
        assert _extract_prompt(sample) == ""


class TestExtractPromptStandalone:
    """独立 prompt 字段"""

    def test_extract_prompt_from_standalone_field(self):
        sample = {"prompt": "Standalone prompt"}
        assert _extract_prompt(sample) == "Standalone prompt"

    def test_extract_prompt_empty_when_nothing(self):
        """没有任何 prompt 来源时返回空字符串"""
        assert _extract_prompt({}) == ""


# ---------- extract_output 测试 ----------

class TestExtractOutput:
    """统一输出提取"""

    def test_from_resps_nested_list(self):
        obj = {"resps": [["output text"]]}
        assert extract_output(obj) == "output text"

    def test_from_resps_flat_list(self):
        obj = {"resps": ["output text"]}
        assert extract_output(obj) == "output text"

    def test_from_filtered_resps(self):
        obj = {"filtered_resps": [["filtered text"]]}
        assert extract_output(obj) == "filtered text"

    def test_empty_resps(self):
        obj = {"resps": []}
        assert extract_output(obj) == ""

    def test_missing_resps_and_filtered(self):
        obj = {}
        assert extract_output(obj) == ""

    def test_empty_string_in_resps(self):
        obj = {"resps": [[""]]}
        assert extract_output(obj) == ""


# ---------- extract_doc_id 测试 ----------

class TestExtractDocId:
    """P1: doc_id=0 修复"""

    def test_doc_id_zero(self):
        """0 是有效的 doc_id"""
        obj = {"doc_id": 0}
        assert extract_doc_id(obj) == 0

    def test_doc_id_none_fallback_to_id(self):
        obj = {"id": 42}
        assert extract_doc_id(obj) == 42

    def test_doc_id_none_id_none_fallback_to_line_num(self):
        obj = {}
        assert extract_doc_id(obj, line_num=5) == 5

    def test_doc_id_zero_not_treated_as_falsy(self):
        """doc_id=0 不应回退到 id 或 line_num"""
        obj = {"doc_id": 0, "id": 99}
        assert extract_doc_id(obj) == 0


# ---------- 空输出测试 ----------

class TestEmptyOutputValidation:
    """P0: 空模型输出必须导致失败"""

    def test_empty_raw_output_fails_validation(self):
        """空输出在 validate_lm_eval_sample_file 中应加入 errors"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False, encoding="utf-8") as f:
            f.write(json.dumps({
                "doc_id": 0,
                "doc": {"problem": "test", "answer": "0"},
                "arguments": {"gen_args_0": {"arg_0": "prompt", "arg_1": {}}},
                "resps": [[""]],
                "filtered_resps": [[""]],
            }) + "\n")
            f.write(json.dumps({
                "doc_id": 1,
                "doc": {"problem": "test2", "answer": "1"},
                "arguments": {"gen_args_0": {"arg_0": "prompt2", "arg_1": {}}},
                "resps": [["valid output"]],
                "filtered_resps": [["valid output"]],
            }) + "\n")
            path = f.name

        try:
            errors = validate_lm_eval_sample_file(path, "test_task", 2, 32768)
            assert any("输出为空" in e for e in errors)
        finally:
            os.unlink(path)

    def test_empty_filtered_output_but_raw_nonempty_passes(self):
        """filtered_output 为空但 raw_output 非空时应通过"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False, encoding="utf-8") as f:
            f.write(json.dumps({
                "doc_id": 0,
                "doc": {"problem": "test", "answer": "0"},
                "arguments": {"gen_args_0": {"arg_0": "prompt", "arg_1": {}}},
                "resps": [["raw output"]],
                "filtered_resps": [[""]],
            }) + "\n")
            path = f.name

        try:
            errors = validate_lm_eval_sample_file(path, "test_task", 1, 32768)
            # resps 非空，不应报 "输出为空"
            assert not any("输出为空" in e for e in errors)
        finally:
            os.unlink(path)

    def test_missing_resps_and_filtered_resps_fails(self):
        """缺少 resps 和 filtered_resps 应失败"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False, encoding="utf-8") as f:
            f.write(json.dumps({
                "doc_id": 0,
                "doc": {"problem": "test", "answer": "0"},
                "arguments": {"gen_args_0": {"arg_0": "prompt", "arg_1": {}}},
            }) + "\n")
            path = f.name

        try:
            errors = validate_lm_eval_sample_file(path, "test_task", 1, 32768)
            assert any("resps" in e for e in errors)
        finally:
            os.unlink(path)


# ---------- doc_id=0 重复检测测试 ----------

class TestDuplicateDocIdZero:
    """P1: doc_id=0 重复检测"""

    def test_duplicate_doc_id_zero_fails(self):
        """两个 doc_id=0 应被检测为重复"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False, encoding="utf-8") as f:
            f.write(json.dumps({
                "doc_id": 0,
                "doc": {"problem": "a", "answer": "0"},
                "arguments": {"gen_args_0": {"arg_0": "p", "arg_1": {}}},
                "resps": [["a"]],
            }) + "\n")
            f.write(json.dumps({
                "doc_id": 0,
                "doc": {"problem": "b", "answer": "1"},
                "arguments": {"gen_args_0": {"arg_0": "p", "arg_1": {}}},
                "resps": [["b"]],
            }) + "\n")
            path = f.name

        try:
            errors = validate_lm_eval_sample_file(path, "test_task", 0, 32768)
            assert any("重复" in e for e in errors)
        finally:
            os.unlink(path)

    def test_unique_doc_ids_including_zero(self):
        """doc_id=0 和 doc_id=1 应不报重复"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False, encoding="utf-8") as f:
            f.write(json.dumps({
                "doc_id": 0,
                "doc": {"problem": "a", "answer": "0"},
                "arguments": {"gen_args_0": {"arg_0": "p", "arg_1": {}}},
                "resps": [["a"]],
            }) + "\n")
            f.write(json.dumps({
                "doc_id": 1,
                "doc": {"problem": "b", "answer": "1"},
                "arguments": {"gen_args_0": {"arg_0": "p", "arg_1": {}}},
                "resps": [["b"]],
            }) + "\n")
            path = f.name

        try:
            errors = validate_lm_eval_sample_file(path, "test_task", 0, 32768)
            assert not any("重复" in e for e in errors)
        finally:
            os.unlink(path)


# ---------- 严格字段验证测试 ----------

class TestValidateExportRecord:
    """P1: 统一 JSONL 严格字段验证"""

    def _valid_record(self):
        return {
            "sample_id": "task:0",
            "task": "local_math500_32k",
            "benchmark": "MATH500",
            "question": "What is 1+1?",
            "prompt": "Problem: 1+1\nAnswer:",
            "raw_output": "The answer is 2.",
            "prompt_hash": "abc123",
            "output_hash": "def456",
            "run_id": "run_test",
            "doc_id": 0,
            "gold_answer": "2",
            "max_gen_toks": 32768,
            "max_model_len": 40960,
            "output_token_count": 10,
            "output_token_count_method": "llama_tokenizer",
        }

    def test_valid_record_passes(self):
        record = self._valid_record()
        errors = validate_export_record(record, 1)
        assert errors == []

    def test_missing_sample_id_fails(self):
        record = self._valid_record()
        record["sample_id"] = ""
        errors = validate_export_record(record, 1)
        assert any("sample_id" in e for e in errors)

    def test_missing_prompt_fails(self):
        record = self._valid_record()
        record["prompt"] = ""
        errors = validate_export_record(record, 1)
        assert any("prompt" in e for e in errors)

    def test_missing_raw_output_fails(self):
        record = self._valid_record()
        record["raw_output"] = ""
        errors = validate_export_record(record, 1)
        assert any("raw_output" in e for e in errors)

    def test_missing_doc_id_fails(self):
        record = self._valid_record()
        del record["doc_id"]
        errors = validate_export_record(record, 1)
        assert any("doc_id" in e for e in errors)

    def test_gold_answer_zero_allowed(self):
        """gold_answer 的值允许是数字 0"""
        record = self._valid_record()
        record["gold_answer"] = 0
        errors = validate_export_record(record, 1)
        # 0 是有效的 gold_answer，不应报错
        assert not any("gold_answer" in e for e in errors)

    def test_gold_answer_string_zero_allowed(self):
        """gold_answer 字符串 "0" 也允许"""
        record = self._valid_record()
        record["gold_answer"] = "0"
        errors = validate_export_record(record, 1)
        assert not any("gold_answer" in e for e in errors)

    def test_max_gen_toks_not_32768_fails(self):
        record = self._valid_record()
        record["max_gen_toks"] = 8192
        errors = validate_export_record(record, 1)
        assert any("max_gen_toks" in e for e in errors)

    def test_max_model_len_too_small_fails(self):
        record = self._valid_record()
        record["max_model_len"] = 16384
        errors = validate_export_record(record, 1)
        assert any("max_model_len" in e for e in errors)

    def test_output_token_count_zero_fails(self):
        record = self._valid_record()
        record["output_token_count"] = 0
        errors = validate_export_record(record, 1)
        assert any("output_token_count" in e for e in errors)

    def test_token_count_method_not_llama_fails(self):
        """token count method 必须为 llama_tokenizer"""
        record = self._valid_record()
        record["output_token_count_method"] = "whitespace"
        errors = validate_export_record(record, 1)
        assert any("output_token_count_method" in e for e in errors)

    def test_missing_question_fails(self):
        record = self._valid_record()
        record["question"] = ""
        errors = validate_export_record(record, 1)
        assert any("question" in e for e in errors)

    def test_missing_run_id_fails(self):
        record = self._valid_record()
        record["run_id"] = ""
        errors = validate_export_record(record, 1)
        assert any("run_id" in e for e in errors)

    def test_missing_hash_fails(self):
        record = self._valid_record()
        record["prompt_hash"] = ""
        errors = validate_export_record(record, 1)
        assert any("prompt_hash" in e for e in errors)


# ---------- validate_exported_generation_file 测试 ----------

class TestValidateExportedFile:
    """验证导出文件整体"""

    def _write_jsonl(self, tmp_path, records):
        path = str(tmp_path / "exported.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path

    def _valid_record(self, doc_id=0):
        return {
            "sample_id": f"task:{doc_id}",
            "task": "local_math500_32k",
            "benchmark": "MATH500",
            "question": "What is 1+1?",
            "prompt": "Problem: 1+1\nAnswer:",
            "raw_output": "The answer is 2.",
            "prompt_hash": "abc123",
            "output_hash": "def456",
            "run_id": "run_test",
            "doc_id": doc_id,
            "gold_answer": "2",
            "max_gen_toks": 32768,
            "max_model_len": 40960,
            "output_token_count": 10,
            "output_token_count_method": "llama_tokenizer",
        }

    def test_valid_file_passes(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            self._valid_record(0),
            self._valid_record(1),
        ])
        errors = validate_exported_generation_file(path, 2, 32768)
        assert errors == []

    def test_wrong_count_fails(self, tmp_path):
        path = self._write_jsonl(tmp_path, [self._valid_record(0)])
        errors = validate_exported_generation_file(path, 2, 32768)
        assert any("行数" in e for e in errors)

    def test_duplicate_sample_id_fails(self, tmp_path):
        path = self._write_jsonl(tmp_path, [
            self._valid_record(0),
            self._valid_record(0),  # 重复
        ])
        errors = validate_exported_generation_file(path, 2, 32768)
        assert any("重复" in e for e in errors)

    def test_empty_file_fails(self, tmp_path):
        path = str(tmp_path / "empty.jsonl")
        with open(path, "w") as f:
            pass
        errors = validate_exported_generation_file(path, 2, 32768)
        assert any("为空" in e for e in errors)

    def test_nonexistent_file_fails(self):
        errors = validate_exported_generation_file("/nonexistent/path.jsonl", 2, 32768)
        assert any("不存在" in e for e in errors)

    def test_corrupt_json_fails(self, tmp_path):
        path = str(tmp_path / "corrupt.jsonl")
        with open(path, "w") as f:
            f.write("{ invalid json }\n")
        errors = validate_exported_generation_file(path, 0, 32768)
        assert any("JSON" in e for e in errors)


# ---------- export_single_task 测试 (使用 mock tokenizer) ----------

class TestExportSingleTask:
    """导出单 task（使用 mock tokenizer）"""

    def _mock_tokenizer(self):
        """创建 mock tokenizer"""
        tok = MagicMock()
        tok.encode = MagicMock(side_effect=lambda text, add_special_tokens=False: text.split())
        return tok

    def _make_sample_file(self, tmp_path, n=2, format_type="dict"):
        """创建 lm-eval sample JSONL"""
        path = str(tmp_path / "samples.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                if format_type == "dict":
                    arguments = {
                        "gen_args_0": {
                            "arg_0": f"Problem: {i}+1\nAnswer:",
                            "arg_1": {"max_gen_toks": 32768}
                        }
                    }
                else:
                    arguments = [[f"Problem: {i}+1\nAnswer:", {"max_gen_toks": 32768}]]
                f.write(json.dumps({
                    "doc_id": i,
                    "doc": {"problem": f"Problem {i}", "answer": str(i)},
                    "arguments": arguments,
                    "resps": [[f"answer {i}"]],
                    "filtered_resps": [[f"answer {i}"]],
                }) + "\n")
        return path

    def _make_task_manifest(self, sample_file, task="local_math500_32k"):
        return {
            "task": task,
            "status": "complete",
            "model_name": "test_model",
            "run_id": "run_test",
            "max_gen_toks": 32768,
            "max_model_len": 40960,
            "temperature": 0.0,
            "do_sample": False,
            "tensor_parallel_size": 1,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 32,
            "gpu_memory_utilization": 0.90,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "base_model": "meta-llama/Llama-3.1-8B",
            "lm_eval_sample_file": sample_file,
            "expected_sample_count": 2,
        }

    def test_export_with_dict_format_prompt(self, tmp_path):
        """P0: lm-eval 0.4.5 dict 格式 prompt 正确提取"""
        sample_file = self._make_sample_file(tmp_path, n=2, format_type="dict")
        manifest = self._make_task_manifest(sample_file)
        out_path = str(tmp_path / "exported.jsonl")
        tokenizer = self._mock_tokenizer()

        result = export_single_task(manifest, "test_model", "/fake/path",
                                     out_path, tokenizer=tokenizer)

        assert result["total_records"] == 2
        # 验证 prompt 被正确提取
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                assert record["prompt"].startswith("Problem:")
                assert record["prompt"] != ""

    def test_export_with_list_format_prompt(self, tmp_path):
        """list 格式 prompt 也正确提取"""
        sample_file = self._make_sample_file(tmp_path, n=2, format_type="list")
        manifest = self._make_task_manifest(sample_file)
        out_path = str(tmp_path / "exported.jsonl")
        tokenizer = self._mock_tokenizer()

        result = export_single_task(manifest, "test_model", "/fake/path",
                                     out_path, tokenizer=tokenizer)
        assert result["total_records"] == 2
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                assert record["prompt"].startswith("Problem:")

    def test_export_rejects_empty_raw_output(self, tmp_path):
        """P0: 空输出导致导出失败"""
        sample_file = str(tmp_path / "samples.jsonl")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "doc_id": 0,
                "doc": {"problem": "Problem 0", "answer": "0"},
                "arguments": {"gen_args_0": {"arg_0": "prompt", "arg_1": {}}},
                "resps": [[""]],
            }) + "\n")

        manifest = self._make_task_manifest(sample_file)
        out_path = str(tmp_path / "exported.jsonl")
        tokenizer = self._mock_tokenizer()

        with pytest.raises(ValueError, match="raw_output 为空"):
            export_single_task(manifest, "test_model", "/fake/path",
                                out_path, tokenizer=tokenizer)

    def test_export_rejects_empty_prompt(self, tmp_path):
        """P0: 空 prompt 导致导出失败"""
        sample_file = str(tmp_path / "samples.jsonl")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "doc_id": 0,
                "doc": {"problem": "Problem 0", "answer": "0"},
                "arguments": {"gen_args_0": {"arg_0": "", "arg_1": {}}},
                "resps": [["valid output"]],
            }) + "\n")

        manifest = self._make_task_manifest(sample_file)
        out_path = str(tmp_path / "exported.jsonl")
        tokenizer = self._mock_tokenizer()

        with pytest.raises(ValueError, match="prompt 为空"):
            export_single_task(manifest, "test_model", "/fake/path",
                                out_path, tokenizer=tokenizer)

    def test_export_rejects_empty_question(self, tmp_path):
        """空 question 导致导出失败"""
        sample_file = str(tmp_path / "samples.jsonl")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "doc_id": 0,
                "doc": {"problem": "", "answer": "0"},
                "arguments": {"gen_args_0": {"arg_0": "prompt", "arg_1": {}}},
                "resps": [["valid output"]],
            }) + "\n")

        manifest = self._make_task_manifest(sample_file)
        out_path = str(tmp_path / "exported.jsonl")
        tokenizer = self._mock_tokenizer()

        with pytest.raises(ValueError, match="question 为空"):
            export_single_task(manifest, "test_model", "/fake/path",
                                out_path, tokenizer=tokenizer)

    def test_export_gold_answer_zero_preserved(self, tmp_path):
        """gold_answer=0 被保留"""
        sample_file = str(tmp_path / "samples.jsonl")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "doc_id": 0,
                "doc": {"problem": "Problem 0", "answer": 0},
                "arguments": {"gen_args_0": {"arg_0": "prompt", "arg_1": {}}},
                "resps": [["valid output"]],
            }) + "\n")

        manifest = self._make_task_manifest(sample_file)
        out_path = str(tmp_path / "exported.jsonl")
        tokenizer = self._mock_tokenizer()

        export_single_task(manifest, "test_model", "/fake/path",
                           out_path, tokenizer=tokenizer)

        with open(out_path, encoding="utf-8") as f:
            record = json.loads(f.readline())
            assert record["gold_answer"] == "0"

    def test_export_token_count_method_is_llama(self, tmp_path):
        """output_token_count_method 必须为 llama_tokenizer"""
        sample_file = self._make_sample_file(tmp_path, n=2)
        manifest = self._make_task_manifest(sample_file)
        out_path = str(tmp_path / "exported.jsonl")
        tokenizer = self._mock_tokenizer()

        export_single_task(manifest, "test_model", "/fake/path",
                           out_path, tokenizer=tokenizer)

        with open(out_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                assert record["output_token_count_method"] == "llama_tokenizer"
                assert record["output_token_count"] > 0

    def test_export_raw_output_complete(self, tmp_path):
        """raw_output 完整保留 CoT"""
        long_output = "Let me think step by step.\n" * 100 + "The answer is 42."
        sample_file = str(tmp_path / "samples.jsonl")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "doc_id": 0,
                "doc": {"problem": "Problem 0", "answer": "42"},
                "arguments": {"gen_args_0": {"arg_0": "prompt", "arg_1": {}}},
                "resps": [[long_output]],
            }) + "\n")

        manifest = self._make_task_manifest(sample_file)
        out_path = str(tmp_path / "exported.jsonl")
        tokenizer = self._mock_tokenizer()

        export_single_task(manifest, "test_model", "/fake/path",
                           out_path, tokenizer=tokenizer)

        with open(out_path, encoding="utf-8") as f:
            record = json.loads(f.readline())
            assert record["raw_output"] == long_output

    def test_export_hash_stable(self, tmp_path):
        """相同输入产生相同 hash"""
        sample_file = self._make_sample_file(tmp_path, n=1)
        manifest = self._make_task_manifest(sample_file)
        out_path = str(tmp_path / "exported.jsonl")
        tokenizer = self._mock_tokenizer()

        export_single_task(manifest, "test_model", "/fake/path",
                           out_path, tokenizer=tokenizer)

        with open(out_path, encoding="utf-8") as f:
            record = json.loads(f.readline())

        # 重新计算 hash 验证
        assert record["prompt_hash"] == _hash(record["prompt"])
        assert record["output_hash"] == _hash(record["raw_output"])

    def test_export_atomic_write(self, tmp_path):
        """原子写入: 写入后无 .tmp 文件残留"""
        sample_file = self._make_sample_file(tmp_path, n=2)
        manifest = self._make_task_manifest(sample_file)
        out_path = str(tmp_path / "exported.jsonl")
        tokenizer = self._mock_tokenizer()

        export_single_task(manifest, "test_model", "/fake/path",
                           out_path, tokenizer=tokenizer)

        assert os.path.isfile(out_path)
        assert not os.path.isfile(out_path + ".tmp")

    def test_export_sample_id_unique(self, tmp_path):
        """sample_id 在文件内唯一"""
        sample_file = self._make_sample_file(tmp_path, n=3)
        manifest = self._make_task_manifest(sample_file)
        out_path = str(tmp_path / "exported.jsonl")
        tokenizer = self._mock_tokenizer()

        export_single_task(manifest, "test_model", "/fake/path",
                           out_path, tokenizer=tokenizer)

        ids = set()
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                assert record["sample_id"] not in ids
                ids.add(record["sample_id"])

    def test_export_duplicate_doc_id_raises(self, tmp_path):
        """重复 doc_id 导致导出失败"""
        sample_file = str(tmp_path / "samples.jsonl")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "doc_id": 0,
                "doc": {"problem": "a", "answer": "0"},
                "arguments": {"gen_args_0": {"arg_0": "p", "arg_1": {}}},
                "resps": [["a"]],
            }) + "\n")
            f.write(json.dumps({
                "doc_id": 0,
                "doc": {"problem": "b", "answer": "1"},
                "arguments": {"gen_args_0": {"arg_0": "p", "arg_1": {}}},
                "resps": [["b"]],
            }) + "\n")

        manifest = self._make_task_manifest(sample_file)
        out_path = str(tmp_path / "exported.jsonl")
        tokenizer = self._mock_tokenizer()

        with pytest.raises(ValueError, match="重复 sample_id"):
            export_single_task(manifest, "test_model", "/fake/path",
                                out_path, tokenizer=tokenizer)

    def test_export_enable_chunked_prefill_in_output(self, tmp_path):
        """导出记录包含 enable_chunked_prefill"""
        sample_file = self._make_sample_file(tmp_path, n=1)
        manifest = self._make_task_manifest(sample_file)
        out_path = str(tmp_path / "exported.jsonl")
        tokenizer = self._mock_tokenizer()

        export_single_task(manifest, "test_model", "/fake/path",
                           out_path, tokenizer=tokenizer)

        with open(out_path, encoding="utf-8") as f:
            record = json.loads(f.readline())
            assert record["enable_chunked_prefill"] is True


# ---------- hash 稳定性测试 ----------

class TestHashStability:
    def test_same_text_same_hash(self):
        assert _hash("hello") == _hash("hello")

    def test_different_text_different_hash(self):
        assert _hash("hello") != _hash("world")

    def test_hash_length(self):
        assert len(_hash("test")) == 16
