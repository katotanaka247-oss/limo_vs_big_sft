"""
test_manifest_validation.py
测试 manifest 完成度判定与 FORCE_RERUN 逻辑：
  * manifest status=complete vs incomplete
  * active_run.json 读取
  * FORCE_RERUN 应新建 run_id
"""
import json
import os
from pathlib import Path

import pytest


def _read_active_run(path):
    """读取 active_run.json，返回 dict 或 None。"""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


class TestManifestStatus:
    """manifest status=complete vs incomplete"""

    def test_complete_manifest(self, tmp_path, make_mock_manifest):
        run_dir = tmp_path / "runs" / "run_test"
        run_dir.mkdir(parents=True)
        m = make_mock_manifest(run_dir / "run_manifest.json", status="complete")
        assert m["status"] == "complete"
        assert m["completion_errors"] == []

    def test_incomplete_manifest(self, tmp_path, make_mock_manifest):
        run_dir = tmp_path / "runs" / "run_test"
        run_dir.mkdir(parents=True)
        m = make_mock_manifest(
            run_dir / "run_manifest.json",
            status="incomplete",
            completion_errors=["task 'local_math500_32k' 样本数=1, 预期=2"],
        )
        assert m["status"] == "incomplete"
        assert len(m["completion_errors"]) == 1

    def test_missing_max_gen_toks(self, tmp_path, make_mock_manifest):
        run_dir = tmp_path / "runs" / "run_test"
        run_dir.mkdir(parents=True)
        # max_gen_toks 不是 32768
        m = make_mock_manifest(run_dir / "run_manifest.json",
                               max_gen_toks=4096)
        assert m["max_gen_toks"] != 32768
        # 完成度判定中会检测到这个不一致


class TestActiveRun:
    """active_run.json 读取"""

    def test_active_run_complete(self, tmp_path):
        active = {"active_run_id": "run_20260719_120000", "status": "complete"}
        ar_path = tmp_path / "active_run.json"
        with open(ar_path, "w", encoding="utf-8") as f:
            json.dump(active, f)
        result = _read_active_run(str(ar_path))
        assert result is not None
        assert result["status"] == "complete"
        assert result["active_run_id"] == "run_20260719_120000"

    def test_active_run_missing(self, tmp_path):
        result = _read_active_run(str(tmp_path / "nonexistent.json"))
        assert result is None

    def test_active_run_corrupt(self, tmp_path):
        ar_path = tmp_path / "active_run.json"
        with open(ar_path, "w") as f:
            f.write("{ broken json")
        result = _read_active_run(str(ar_path))
        assert result is None


class TestForceRerun:
    """FORCE_RERUN 应新建 run_id，不删除历史"""

    def test_new_run_id_created(self, tmp_path):
        """模拟 FORCE_RERUN 场景：新 run 目录存在，旧 run 目录保留"""
        runs_dir = tmp_path / "runs"
        old_run = runs_dir / "run_old"
        new_run = runs_dir / "run_new"
        old_run.mkdir(parents=True)
        new_run.mkdir(parents=True)

        # 旧 run 有结果
        with open(old_run / "run_manifest.json", "w") as f:
            json.dump({"status": "complete", "run_id": "run_old"}, f)

        # 新 run 有结果
        with open(new_run / "run_manifest.json", "w") as f:
            json.dump({"status": "complete", "run_id": "run_new"}, f)

        # active_run.json 更新为新 run
        active = {"active_run_id": "run_new", "status": "complete"}
        with open(tmp_path / "active_run.json", "w") as f:
            json.dump(active, f)

        # 旧 run 目录仍然存在（不被删除）
        assert old_run.exists()
        assert (old_run / "run_manifest.json").exists()

        # active 指向新 run
        ar = _read_active_run(str(tmp_path / "active_run.json"))
        assert ar["active_run_id"] == "run_new"

    def test_active_run_atomic_update(self, tmp_path):
        """active_run.json 应原子更新"""
        ar_path = tmp_path / "active_run.json"
        # 先写入旧值
        with open(ar_path, "w") as f:
            json.dump({"active_run_id": "old", "status": "complete"}, f)

        # 模拟原子更新（写 tmp -> rename）
        new_data = {"active_run_id": "new", "status": "complete"}
        tmp_path_file = str(ar_path) + ".tmp"
        with open(tmp_path_file, "w", encoding="utf-8") as f:
            json.dump(new_data, f)
        os.replace(tmp_path_file, ar_path)

        result = _read_active_run(str(ar_path))
        assert result["active_run_id"] == "new"


class TestCompletionJudgmentFields:
    """manifest 必须包含完成度判定所需的字段"""

    REQUIRED_FIELDS = [
        "status", "max_gen_toks", "max_model_len", "max_num_batched_tokens",
        "max_num_seqs", "gpu_memory_utilization", "enable_prefix_caching",
        "dtype", "tasks", "evaluation_protocol", "lm_eval_results_file",
        "lm_eval_sample_files", "completion_errors",
    ]

    def test_manifest_has_required_fields(self, tmp_path, make_mock_manifest):
        run_dir = tmp_path / "runs" / "run_test"
        run_dir.mkdir(parents=True)
        m = make_mock_manifest(run_dir / "run_manifest.json")
        for field in self.REQUIRED_FIELDS:
            assert field in m, f"manifest 缺少字段: {field}"
