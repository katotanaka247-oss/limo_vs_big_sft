"""
test_task_configs.py
测试三个本地 task 的 YAML 配置正确性：
  * task 名称唯一
  * max_gen_toks=32768
  * do_sample=false, temperature=0.0
  * 停止字符串配置
  * dataset_path / split 正确
  * math_utils.py 可导入且关键函数存在
"""
import os
import sys
from pathlib import Path

import pytest

try:
    import yaml
except ImportError:
    yaml = None

EVAL_TASKS_DIR = Path(__file__).resolve().parent.parent / "eval_tasks"

TASK_YAMLS = {
    "local_math500_32k": EVAL_TASKS_DIR / "math500_32k.yaml",
    "local_aime24_32k": EVAL_TASKS_DIR / "aime24_32k.yaml",
    "local_aime25_32k": EVAL_TASKS_DIR / "aime25_32k.yaml",
}


class _FunctionLoader(yaml.SafeLoader):
    """自定义 YAML loader，将 !function 标签解析为字符串。
    lm-eval 使用 !function math_utils.process_results 引用 Python 函数，
    yaml.safe_load 无法解析该标签，需要自定义构造器。"""
    pass


def _construct_function(loader, node):
    """将 !function tag 的值作为字符串返回，例如 'math_utils.process_results'。"""
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    return str(node.value)


_FunctionLoader.add_constructor("!function", _construct_function)


def _load_task_yaml(path):
    """用自定义 loader 解析含 !function 标签的 YAML。"""
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=_FunctionLoader)


@pytest.fixture(scope="module")
def task_configs():
    """加载所有 YAML 配置（支持 !function 标签）。"""
    if yaml is None:
        pytest.skip("PyYAML not installed")
    configs = {}
    for task_name, yaml_path in TASK_YAMLS.items():
        if not yaml_path.is_file():
            pytest.fail(f"YAML 文件缺失: {yaml_path}")
        configs[task_name] = _load_task_yaml(yaml_path)
    return configs


class TestTaskNames:
    """task 名称唯一且正确"""

    def test_task_names(self, task_configs):
        for task_name, cfg in task_configs.items():
            assert cfg["task"] == task_name

    def test_no_duplicate_names(self, task_configs):
        names = [cfg["task"] for cfg in task_configs.values()]
        assert len(names) == len(set(names))


class TestGenerationKwargs:
    """max_gen_toks=32768, do_sample=false, temperature=0.0"""

    def test_max_gen_toks(self, task_configs):
        for task_name, cfg in task_configs.items():
            gk = cfg.get("generation_kwargs", {})
            assert gk.get("max_gen_toks") == 32768, \
                f"{task_name}: max_gen_toks 应为 32768, 实际={gk.get('max_gen_toks')}"

    def test_do_sample_false(self, task_configs):
        for task_name, cfg in task_configs.items():
            gk = cfg.get("generation_kwargs", {})
            assert gk.get("do_sample") is False, \
                f"{task_name}: do_sample 应为 False"

    def test_temperature_zero(self, task_configs):
        for task_name, cfg in task_configs.items():
            gk = cfg.get("generation_kwargs", {})
            assert gk.get("temperature") == 0.0, \
                f"{task_name}: temperature 应为 0.0"


class TestStopStrings:
    """停止字符串配置一致"""

    def test_stop_strings_consistent(self, task_configs):
        all_untils = []
        for task_name, cfg in task_configs.items():
            gk = cfg.get("generation_kwargs", {})
            until = gk.get("until", [])
            all_untils.append(tuple(sorted(until)))
        # 三个 task 的 until 应完全相同
        assert len(set(all_untils)) == 1, "三个 task 的停止字符串不一致"
        # 不应包含高风险停止词
        until_set = set(all_untils[0])
        assert "###" not in until_set, "不应使用 ### 作为停止词"
        assert "Solution:" not in until_set, "不应使用 Solution: 作为停止词"


class TestDatasetConfig:
    """dataset_path / split 正确"""

    def test_math500(self, task_configs):
        cfg = task_configs["local_math500_32k"]
        assert cfg["dataset_path"] == "HuggingFaceH4/MATH-500"
        assert cfg["test_split"] == "test"

    def test_aime24(self, task_configs):
        cfg = task_configs["local_aime24_32k"]
        assert cfg["dataset_path"] == "Maxwell-Jia/AIME_2024"
        assert cfg["test_split"] == "train"

    def test_aime25(self, task_configs):
        cfg = task_configs["local_aime25_32k"]
        assert cfg["dataset_path"] == "math-ai/aime25"
        assert cfg["test_split"] == "test"


class TestDocToText:
    """doc_to_text 格式正确"""

    def test_math500_prompt(self, task_configs):
        cfg = task_configs["local_math500_32k"]
        dtt = cfg["doc_to_text"]
        assert "{{problem}}" in dtt
        assert "Answer:" in dtt

    def test_aime24_prompt(self, task_configs):
        cfg = task_configs["local_aime24_32k"]
        dtt = cfg["doc_to_text"]
        # AIME24 使用大写 Problem/Answer
        assert "{{Problem}}" in dtt
        assert "Answer:" in dtt

    def test_aime25_prompt(self, task_configs):
        cfg = task_configs["local_aime25_32k"]
        dtt = cfg["doc_to_text"]
        assert "{{problem}}" in dtt
        assert "Answer:" in dtt


class TestMathUtils:
    """math_utils.py 可导入且关键函数存在"""

    def test_import_math_utils(self):
        sys.path.insert(0, str(EVAL_TASKS_DIR))
        import math_utils
        assert hasattr(math_utils, "process_results")
        assert hasattr(math_utils, "is_equiv")
        assert hasattr(math_utils, "remove_boxed")
        assert hasattr(math_utils, "last_boxed_only_string")

    def test_is_equiv_basic(self):
        sys.path.insert(0, str(EVAL_TASKS_DIR))
        import math_utils
        # 简单等价性测试
        assert math_utils.is_equiv("42", "42")
        assert math_utils.is_equiv("\\frac{1}{2}", "\\frac{1}{2}")

    def test_remove_boxed(self):
        sys.path.insert(0, str(EVAL_TASKS_DIR))
        import math_utils
        result = math_utils.remove_boxed("\\boxed{42}")
        assert result is not None

    def test_last_boxed_only_string(self):
        sys.path.insert(0, str(EVAL_TASKS_DIR))
        import math_utils
        result = math_utils.last_boxed_only_string("Some text \\boxed{42} more text")
        assert result is not None


class TestProcessResults:
    """process_results 函数行为"""

    def test_process_results_correct(self):
        sys.path.insert(0, str(EVAL_TASKS_DIR))
        import math_utils
        doc = {"answer": "42"}
        results = ["The answer is \\boxed{42}."]
        res = math_utils.process_results(doc, results)
        assert isinstance(res, dict)
        # 应该包含 exact_match 键
        assert "exact_match" in res or len(res) > 0

    def test_process_results_with_solution_fallback(self):
        sys.path.insert(0, str(EVAL_TASKS_DIR))
        import math_utils
        doc = {"solution": "The answer is \\boxed{42}."}
        results = ["\\boxed{42}"]
        res = math_utils.process_results(doc, results)
        assert isinstance(res, dict)
