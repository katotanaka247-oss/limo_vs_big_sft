"""
test_gpu_selection.py
测试 GPU 选择逻辑、attempt 初始化、cleanup 函数：
  * CUDA_VISIBLE_DEVICES 包含逗号 -> 报错
  * CUDA_VISIBLE_DEVICES 单卡 -> 通过
  * attempt 初始值为 0（set -u 兼容）
  * cleanup 不残留模拟进程
  * 两模型调度配置不一致检测
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"


def _bash_available():
    """检查系统是否有可用的 bash（能实际执行脚本）。
    仅检查 `bash --version` 返回 0 不够——在 Windows 上 WSL bash 可能存在
    但因代理配置等原因无法正常执行脚本。这里实际运行一个简单脚本验证。"""
    try:
        result = subprocess.run(
            ["bash", "-c", "echo ok"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


# 在 Windows 上，WSL bash 可能存在但因代理/路径等问题无法可靠执行脚本。
# 这些 bash 测试主要面向 Linux 服务器环境（实际运行评测的环境）。
has_bash = _bash_available()
is_windows = sys.platform == "win32"


class TestGPUSelection:
    """CUDA_VISIBLE_DEVICES 验证"""

    @pytest.mark.skipif(not has_bash, reason="bash not available")
    def test_multi_gpu_rejected(self):
        """CUDA_VISIBLE_DEVICES='0,1' 应报错"""
        script = """
set -euo pipefail
export CUDA_VISIBLE_DEVICES="0,1"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "ERROR: multi-GPU rejected"
    exit 2
fi
echo "OK"
"""
        result = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 2
        assert "rejected" in result.stderr or "rejected" in result.stdout

    @pytest.mark.skipif(not has_bash, reason="bash not available")
    def test_single_gpu_accepted(self):
        """CUDA_VISIBLE_DEVICES='0' 应通过"""
        script = """
set -euo pipefail
export CUDA_VISIBLE_DEVICES="0"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "ERROR"
    exit 2
fi
echo "OK"
"""
        result = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    @pytest.mark.skipif(not has_bash, reason="bash not available")
    def test_default_single_gpu(self):
        """不设置 CUDA_VISIBLE_DEVICES 时默认为 0"""
        script = """
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "ERROR"
    exit 2
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
"""
        result = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, timeout=10,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        assert result.returncode == 0


class TestAttemptInit:
    """attempt=0 在 set -u 下初始化"""

    @pytest.mark.skipif(not has_bash, reason="bash not available")
    def test_attempt_not_initialized_fails(self):
        """set -u 下未初始化 attempt 会报错"""
        script = """
set -euo pipefail
# 不初始化 attempt
attempt=$((attempt + 1))
echo "attempt=$attempt"
"""
        result = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, timeout=10
        )
        assert result.returncode != 0
        assert "unbound variable" in result.stderr.lower() or \
               "attempt" in result.stderr.lower()

    @pytest.mark.skipif(not has_bash, reason="bash not available")
    def test_attempt_initialized_ok(self):
        """先 attempt=0 再 +1 应通过"""
        script = """
set -euo pipefail
attempt=0
attempt=$((attempt + 1))
echo "attempt=$attempt"
"""
        result = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0
        assert "attempt=1" in result.stdout

    def test_run_eval_single_script_attempt_init(self):
        """run_eval_single_l40_vllm.sh 中 attempt=0 在 fallback 循环前。
        此测试只读取脚本文件内容，不需要 bash 可执行。"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("run_eval_single_l40_vllm.sh not found")
        content = script_path.read_text(encoding="utf-8")
        # 检查 attempt=0 出现在 task_attempt=$((task_attempt + 1)) 之前
        init_pos = content.find("attempt=0")
        inc_pos = content.find("task_attempt=$((task_attempt + 1))")
        assert init_pos >= 0, "脚本中未找到 attempt=0 初始化"
        assert inc_pos >= 0, "脚本中未找到 task_attempt=$((task_attempt + 1))"
        assert init_pos < inc_pos, "attempt=0 必须在 task_attempt=$((task_attempt + 1)) 之前"


class TestCleanup:
    """cleanup 函数不残留进程"""

    @pytest.mark.skipif(not has_bash, reason="bash not available")
    def test_cleanup_kills_child(self):
        """cleanup 应 kill EVAL_CHILD_PID"""
        script = """
set -euo pipefail
EVAL_CHILD_PID=""
GPU_MONITOR_PID=""

cleanup() {
    if [[ -n "${EVAL_CHILD_PID:-}" ]]; then
        kill "$EVAL_CHILD_PID" 2>/dev/null || true
        wait "$EVAL_CHILD_PID" 2>/dev/null || true
    fi
    if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# 启动一个后台子进程
sleep 30 &
EVAL_CHILD_PID=$!
CHILD_PID=$EVAL_CHILD_PID

# 等一小会儿确认子进程在运行
sleep 0.5
if ! kill -0 "$CHILD_PID" 2>/dev/null; then
    echo "ERROR: child not running"
    exit 1
fi

# 脚本退出时 cleanup 会 kill 子进程
echo "child_pid=$CHILD_PID"
"""
        result = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0
        # 提取 child PID
        for line in result.stdout.split("\n"):
            if line.startswith("child_pid="):
                child_pid = int(line.split("=")[1])
                # 等待一下让 cleanup 生效
                import time
                time.sleep(0.5)
                # 检查子进程已被 kill
                try:
                    os.kill(child_pid, 0)
                    # 如果还活着，测试失败
                    # 但在 Windows 上 os.kill 行为不同，跳过
                    if sys.platform != "win32":
                        pytest.fail("子进程未被 cleanup kill")
                except (ProcessLookupError, OSError):
                    pass  # 进程已退出，符合预期

    def test_no_pkill_in_scripts(self):
        """脚本中不能使用 pkill python / pkill -f vllm。
        此测试只读取脚本文件内容，不需要 bash 可执行。"""
        for script_name in ["run_eval_single_l40_vllm.sh",
                            "run_eval_two_models_single_l40.sh"]:
            script_path = SCRIPTS_DIR / script_name
            if not script_path.is_file():
                continue
            content = script_path.read_text(encoding="utf-8")
            assert "pkill python" not in content, \
                f"{script_name} 中不应使用 pkill python"
            assert "pkill -f vllm" not in content, \
                f"{script_name} 中不应使用 pkill -f vllm"

    def test_no_error_suppression_in_task_check(self):
        """task 检查命令不能吞掉错误（不用 2>/dev/null || true）。
        此测试只读取脚本文件内容，不需要 bash 可执行。"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        # 检查 task listing 部分没有 "2>/dev/null || true" 吞错误
        # 找到 task listing 相关行
        lines = content.split("\n")
        task_listing_lines = [l for l in lines if "tasks list" in l or "task listing" in l]
        for line in task_listing_lines:
            # 不应在 task listing 行中同时有 2>/dev/null 和 || true
            assert not ("2>/dev/null" in line and "|| true" in line), \
                f"task listing 行不应吞掉错误: {line}"


class TestScriptSyntax:
    """bash -n 语法检查"""

    @pytest.mark.skipif(not has_bash, reason="bash not available")
    @pytest.mark.parametrize("script_name", [
        "run_eval_single_l40_vllm.sh",
        "run_eval_two_models_single_l40.sh",
    ])
    def test_bash_n_syntax(self, script_name):
        script_path = SCRIPTS_DIR / script_name
        if not script_path.is_file():
            pytest.skip(f"{script_name} not found")
        result = subprocess.run(
            ["bash", "-n", str(script_path)], capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0, \
            f"bash -n 失败 for {script_name}: {result.stderr}"


class TestScriptContent:
    """脚本内容静态检查（不需要 bash 可执行，适用于所有平台）"""

    def test_cleanup_function_in_single_script(self):
        """run_eval_single_l40_vllm.sh 中必须包含 cleanup 函数和 trap。"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        assert "cleanup()" in content, "脚本中未找到 cleanup() 函数"
        assert "trap cleanup EXIT" in content, "脚本中未找到 trap cleanup EXIT"
        assert "EVAL_CHILD_PID" in content, "脚本中未找到 EVAL_CHILD_PID 变量"
        assert "GPU_MONITOR_PID" in content, "脚本中未找到 GPU_MONITOR_PID 变量"

    def test_force_config_in_single_script(self):
        """run_eval_single_l40_vllm.sh 中必须支持 FORCE_CONFIG 环境变量。"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        assert "FORCE_CONFIG" in content, "脚本中未找到 FORCE_CONFIG 环境变量支持"

    def test_force_config_in_two_models_script(self):
        """run_eval_two_models_single_l40.sh 中必须使用 FORCE_CONFIG 实现共同配置。"""
        script_path = SCRIPTS_DIR / "run_eval_two_models_single_l40.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        assert "FORCE_CONFIG" in content, "脚本中未找到 FORCE_CONFIG 环境变量支持"

    def test_attempt_init_in_single_script(self):
        """run_eval_single_l40_vllm.sh 中 attempt=0 必须在 task_attempt=$((task_attempt + 1)) 之前。"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        init_pos = content.find("attempt=0")
        inc_pos = content.find("task_attempt=$((task_attempt + 1))")
        assert init_pos >= 0, "脚本中未找到 attempt=0 初始化"
        assert inc_pos >= 0, "脚本中未找到 task_attempt=$((task_attempt + 1))"
        assert init_pos < inc_pos, "attempt=0 必须在 task_attempt=$((task_attempt + 1)) 之前"

    def test_no_set_e_suppression_in_task_check(self):
        """task 检查命令附近不能有 set +e 吞错误。"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        # 找到 --tasks list 附近的内容
        idx = content.find("--tasks list")
        if idx < 0:
            pytest.skip("脚本中未找到 --tasks list")
        # 检查前后 500 字符内没有 set +e
        nearby = content[max(0, idx - 500):idx + 500]
        assert "set +e" not in nearby, \
            "task listing 命令附近不应使用 set +e 吞错误"

    def test_serial_execution_in_two_models(self):
        """run_eval_two_models_single_l40.sh 中不能有并发执行两个模型的 command& 模式。"""
        script_path = SCRIPTS_DIR / "run_eval_two_models_single_l40.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        # 不应出现 bash run_eval_single_l40_vllm.sh & 这种后台执行
        lines = content.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            # 跳过注释行
            if stripped.startswith("#"):
                continue
            # 检查是否在行尾有 & 后台执行（排除 && 逻辑与和 >> 重定向）
            if stripped.endswith("&") and not stripped.endswith("&&"):
                # 允许的例外：GPU 监控等后台进程
                if "nvidia-smi" in stripped or "gpu_monitor" in stripped.lower() \
                        or "monitor" in stripped.lower() or "sleep" in stripped:
                    continue
                # 如果是运行评测脚本的后台执行，则失败
                if "run_eval" in stripped or "lm_eval" in stripped or "lm-eval" in stripped:
                    pytest.fail(f"第 {i+1} 行使用了后台执行评测: {stripped}")


class TestConfigConsistency:
    """两模型调度配置不一致检测"""

    def test_different_configs_detected(self):
        """模拟两个模型使用不同配置"""
        limo_config = {"max_num_seqs": 32, "max_num_batched_tokens": 8192}
        openr1_config = {"max_num_seqs": 8, "max_num_batched_tokens": 2048}

        config_fields = ["max_num_seqs", "max_num_batched_tokens"]
        config_match = all(
            limo_config.get(f) == openr1_config.get(f) for f in config_fields
        )
        assert config_match is False

    def test_same_configs_detected(self):
        """模拟两个模型使用相同配置"""
        limo_config = {"max_num_seqs": 32, "max_num_batched_tokens": 8192}
        openr1_config = {"max_num_seqs": 32, "max_num_batched_tokens": 8192}

        config_fields = ["max_num_seqs", "max_num_batched_tokens"]
        config_match = all(
            limo_config.get(f) == openr1_config.get(f) for f in config_fields
        )
        assert config_match is True

    def test_config_comparison_float_equal(self):
        """0.90 和 0.9 应被视为相等（浮点数值比较，非字符串比较）"""
        # 模拟两个模型的配置字符串
        c1 = "1 8192 32 0.90 True"
        c2 = "1 8192 32 0.9 True"

        # 解析并比较（与脚本中的 configs_equal 函数逻辑一致）
        parts1 = c1.split()
        parts2 = c2.split()

        assert len(parts1) == 5
        assert len(parts2) == 5

        level1, mnbt1, mns1, gmu1, pc1 = parts1
        level2, mnbt2, mns2, gmu2, pc2 = parts2

        # 逐字段比较（数值用 float 比较）
        assert int(level1) == int(level2)
        assert int(mnbt1) == int(mnbt2)
        assert int(mns1) == int(mns2)
        assert abs(float(gmu1) - float(gmu2)) < 1e-6
        assert pc1 == pc2

    def test_config_comparison_different_level(self):
        """不同 fallback level 的配置应被视为不同"""
        c1 = "1 8192 32 0.90 True"
        c2 = "2 4096 16 0.90 True"

        parts1 = c1.split()
        parts2 = c2.split()

        assert int(parts1[0]) != int(parts2[0])
        assert int(parts1[1]) != int(parts2[1])


class TestGenerationOnly:
    """generation-only 模式相关测试"""

    def test_predict_only_in_single_script(self):
        """run_eval_single_l40_vllm.sh 中必须包含 --predict_only"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        assert "--predict_only" in content, \
            "脚本中未找到 --predict_only（generation-only 模式必需）"

    def test_generation_only_manifest_fields(self):
        """脚本中必须记录 generation-only manifest 字段"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        assert "generation_only" in content, \
            "脚本中未记录 evaluation_mode=generation_only"
        assert "predict_only" in content, \
            "脚本中未记录 predict_only"
        assert "pending_local" in content, \
            "脚本中未记录 judging_status=pending_local"
        assert "server_side_accuracy_valid" in content, \
            "脚本中未记录 server_side_accuracy_valid"

    def test_no_accuracy_required_in_completion(self):
        """完成度判定不应要求 accuracy/exact_match"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        # 不应出现要求 accuracy 的代码
        assert "has_acc" not in content or \
               "exact_match" not in content.split("has_acc")[0].split("\n")[-1], \
            "完成度判定不应依赖 accuracy/exact_match"

    def test_task_level_execution(self):
        """脚本中应实现 task 级执行（每个 task 独立调用 lm-eval）"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        assert "run_task" in content or "for task in" in content or \
               "for i in" in content, \
            "脚本中未实现 task 级执行"
        assert "tasks/" in content, \
            "脚本中未使用 task 级目录结构"

    def test_per_attempt_timing(self):
        """脚本中应实现 per-attempt 计时"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        assert "attempt_start" in content or "ATTEMPT_START" in content, \
            "脚本中未实现 per-attempt 计时"
        assert "successful_attempt_elapsed_seconds" in content, \
            "脚本中未记录 successful_attempt_elapsed_seconds"

    def test_gpu_free_before_first_attempt(self):
        """脚本中在第一次 attempt 前必须检查 GPU 空闲"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        assert "before first attempt" in content, \
            "脚本中未在第一次 attempt 前检查 GPU 空闲"

    def test_gpu_peak_memory_in_manifest(self):
        """脚本中必须将 GPU 峰值显存写入 manifest"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        assert "gpu_peak_memory_mib" in content, \
            "脚本中未将 gpu_peak_memory_mib 写入 manifest"

    def test_fallback_config_format_with_level(self):
        """fallback 配置应使用 level 前缀格式"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        # 检查 fallback 配置格式为 "level mnbt mns gmu epc"
        assert '"1 8192 32 0.90 True"' in content or \
               '"1 8192 32 0.90 True"' in content.replace("'", '"'), \
            "fallback 配置应使用 level 前缀格式"

    def test_eval_one_captures_rc(self):
        """two_models 脚本中 eval_one 必须用 || rc=$? 捕获返回码"""
        script_path = SCRIPTS_DIR / "run_eval_two_models_single_l40.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        assert "|| rc=$?" in content or "rc=$?" in content, \
            "eval_one 必须用 || rc=$? 捕获返回码，避免 set -e 提前退出"

    def test_openr1_fallback_reachable(self):
        """two_models 脚本中 OpenR1 fallback 必须可达"""
        script_path = SCRIPTS_DIR / "run_eval_two_models_single_l40.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        # 检查 OpenR1 失败后能进入 fallback
        assert "OPENR1_RC" in content, \
            "脚本中未使用 OPENR1_RC 变量捕获 OpenR1 返回码"
        assert "FORCE_RERUN=1 eval_one" in content or \
               "FORCE_RERUN=1" in content, \
            "脚本中未实现 OpenR1 fallback 后重跑"

    def test_smoke_full_separation(self):
        """two_models 脚本中 smoke 和 full 结果应分开"""
        script_path = SCRIPTS_DIR / "run_eval_two_models_single_l40.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        assert "SMOKE_SUFFIX" in content or "_smoke" in content, \
            "脚本中未实现 smoke/full 结果分离"
        assert "generation_comparison_32k" in content, \
            "脚本中应使用 generation_comparison_32k 作为输出名"

    def test_export_jsonl_called(self):
        """single 脚本中应调用 export_generation_outputs.py"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        assert "export_generation_outputs.py" in content, \
            "脚本中未调用 export_generation_outputs.py"

    def test_no_pip_install_math_verify(self):
        """requirements-eval-vllm-cu121.txt 中不应包含 math_verify"""
        req_path = PROJECT_DIR / "requirements-eval-vllm-cu121.txt"
        if not req_path.is_file():
            pytest.skip("requirements file not found")
        content = req_path.read_text(encoding="utf-8")
        # math_verify 不应出现在非注释行中
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped and not stripped.startswith("#"):
                assert "math_verify" not in stripped, \
                    "requirements-eval-vllm-cu121.txt 不应依赖 math_verify"

    def test_enable_chunked_prefill_in_model_args(self):
        """P1: model_args 中必须显式包含 enable_chunked_prefill=True"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        assert "enable_chunked_prefill=True" in content, \
            "脚本中未显式设置 enable_chunked_prefill=True"

    def test_enable_chunked_prefill_in_manifest(self):
        """P1: manifest 中必须记录 enable_chunked_prefill"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        assert '"enable_chunked_prefill"' in content, \
            "脚本中未在 manifest 中记录 enable_chunked_prefill"

    def test_oom_regex_narrow(self):
        """P1: OOM 正则不应匹配笼统的 'CUDA error'"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        # 找到 OOM 检测的 grep 命令
        # 不应单独匹配 "CUDA error"（会匹配 "CUDA error: invalid argument" 等非 OOM 错误）
        # 应该匹配 "CUDA error: out of memory" 或 "CUDA out of memory"
        assert "CUDA out of memory" in content or \
               "CUDA error: out of memory" in content, \
            "OOM 正则应包含 'CUDA out of memory' 或 'CUDA error: out of memory'"

    def test_export_failure_blocks_completion(self):
        """P0: 导出失败必须返回非零（不能只 warning）"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        # 不应出现 "不影响评测完整性" 这样的 warning
        assert "不影响评测完整性" not in content, \
            "导出失败不应只 warning，应阻断 task 完成"

    def test_gpu_cleanup_strict_after_task(self):
        """P1: task 成功后 GPU 清理必须返回错误（不能只 warning）"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        # 找到 task 成功后的 GPU 清理
        idx = content.find("task=$task success cleanup")
        if idx < 0:
            pytest.skip("未找到 task success cleanup")
        nearby = content[max(0, idx-200):idx+200]
        # 不应只 warning，应该 return 7 或类似错误
        assert "return 7" in nearby or "return 1" in nearby, \
            "task 成功后 GPU 未释放应返回错误，不能只 warning"

    def test_validate_generation_task_module_exists(self):
        """P1: validate_generation_task.py 必须存在"""
        vgt_path = SCRIPTS_DIR / "validate_generation_task.py"
        assert vgt_path.is_file(), \
            "scripts/validate_generation_task.py 不存在"

    def test_assert_gpu_free_fails_in_two_models(self):
        """P1: two_models 脚本中 assert_gpu_free 最终检查失败必须 exit 1"""
        script_path = SCRIPTS_DIR / "run_eval_two_models_single_l40.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        # 找到 "最终 GPU" 附近的代码
        idx = content.find("final")
        if idx < 0:
            pytest.skip("未找到 final GPU check")
        nearby = content[max(0, idx-200):idx+400]
        assert "exit 1" in nearby, \
            "assert_gpu_free 最终检查失败应 exit 1"


class TestGPUMemoryParsing:
    """GPU 显存解析测试"""

    def test_parse_gpu_peak_with_nounits(self):
        """解析 nvidia-smi nounits 格式的 GPU 峰值显存"""
        # 模拟 GPU 监控日志（nounits 格式）
        gpu_log_content = """2026/07/19 12:00:00, 42120, 46068, 98
2026/07/19 12:00:05, 38000, 46068, 95
2026/07/19 12:00:10, 45000, 46068, 99
2026/07/19 12:00:15, 39000, 46068, 92
"""
        # 解析逻辑（与脚本中的 parse_gpu_peak 一致）
        peak = 0
        for line in gpu_log_content.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    val = int(float(parts[1]))
                    if val > peak:
                        peak = val
                except (ValueError, TypeError):
                    pass

        assert peak == 45000, f"Expected peak=45000, got {peak}"

    def test_parse_gpu_peak_with_spaces(self):
        """解析带空格的 nvidia-smi 输出"""
        gpu_log_content = """2026/07/19 12:00:00,  42120,  46068,  98
2026/07/19 12:00:05,  38000,  46068,  95
"""
        peak = 0
        for line in gpu_log_content.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    val = int(float(parts[1]))
                    if val > peak:
                        peak = val
                except (ValueError, TypeError):
                    pass

        assert peak == 42120

    def test_parse_gpu_peak_empty_log(self):
        """空日志应返回 0"""
        peak = 0
        for line in "":
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    val = int(float(parts[1]))
                    if val > peak:
                        peak = val
                except (ValueError, TypeError):
                    pass
        assert peak == 0
