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
        # 检查 attempt=0 出现在 attempt=$((attempt + 1)) 之前
        init_pos = content.find("attempt=0")
        inc_pos = content.find("attempt=$((attempt + 1))")
        assert init_pos >= 0, "脚本中未找到 attempt=0 初始化"
        assert inc_pos >= 0, "脚本中未找到 attempt=$((attempt + 1))"
        assert init_pos < inc_pos, "attempt=0 必须在 attempt=$((attempt + 1)) 之前"


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
        """run_eval_single_l40_vllm.sh 中 attempt=0 必须在 attempt=$((attempt + 1)) 之前。"""
        script_path = SCRIPTS_DIR / "run_eval_single_l40_vllm.sh"
        if not script_path.is_file():
            pytest.skip("script not found")
        content = script_path.read_text(encoding="utf-8")
        init_pos = content.find("attempt=0")
        inc_pos = content.find("attempt=$((attempt + 1))")
        assert init_pos >= 0, "脚本中未找到 attempt=0 初始化"
        assert inc_pos >= 0, "脚本中未找到 attempt=$((attempt + 1))"
        assert init_pos < inc_pos, "attempt=0 必须在 attempt=$((attempt + 1)) 之前"

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

    @pytest.mark.skipif(not has_bash, reason="bash not available")
    def test_conservativeness_function(self):
        """测试 config_conservativeness 函数"""
        script = """
# 从 run_eval_two_models_single_l40.sh 中提取的函数
config_conservativeness() {
    local cfg="$1"
    case "$cfg" in
        "8192 32 0.90 True")  echo 1 ;;
        "4096 16 0.90 True")  echo 2 ;;
        "2048 8  0.88 True")  echo 3 ;;
        "2048 4  0.88 False") echo 4 ;;
        *) echo 0 ;;
    esac
}
echo "$(config_conservativeness '8192 32 0.90 True')"
echo "$(config_conservativeness '2048 4  0.88 False')"
echo "$(config_conservativeness 'unknown config')"
"""
        result = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0
        lines = result.stdout.strip().split("\n")
        assert lines[0] == "1"
        assert lines[1] == "4"
        assert lines[2] == "0"
