"""
diagnose_eval_environment.py
全面诊断评测环境，任何关键检查失败时返回非零退出码。

检查项:
  * Python / Torch / Torch CUDA / CUDA available / L40
  * Transformers / vLLM / lm-eval / PEFT / datasets / safetensors
  * GPU compute capability
  * merged model 完整性（两个模型）
  * 三个本地 task（eval_tasks/ 目录）
  * Hugging Face 数据集可访问性
  * 磁盘剩余空间
  * CPU 内存
  * 目标 GPU 是否空闲

输出: results/environment_diagnostic.json
"""
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
EVAL_TASKS_DIR = PROJECT_DIR / "eval_tasks"
REQUIRED_TASKS = ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]
MERGED_DIRS = [
    ("LIMO-817", PROJECT_DIR / "outputs" / "llama31_8b_limo_817_merged"),
    ("OpenR1-10K", PROJECT_DIR / "outputs" / "llama31_8b_openr1_10k_merged"),
]


def _ok(checks, name, detail=None):
    checks.append({"name": name, "status": "ok", "detail": detail or {}})


def _fail(checks, name, detail=None):
    checks.append({"name": name, "status": "fail", "detail": detail or {}})


def _warn(checks, name, detail=None):
    checks.append({"name": name, "status": "warn", "detail": detail or {}})


def _version(pkg_name):
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(pkg_name)
        except PackageNotFoundError:
            return None
    except Exception:
        return None


def check_python(checks):
    v = sys.version.split()[0]
    major, minor = sys.version_info[:2]
    if major == 3 and minor == 10:
        _ok(checks, "python", {"version": v})
    else:
        _fail(checks, "python", {"version": v, "expected": "3.10.x"})


def check_torch(checks):
    try:
        import torch
        v = torch.__version__
        cuda_v = torch.version.cuda
        available = torch.cuda.is_available()
        detail = {
            "version": v,
            "cuda_version": cuda_v,
            "cuda_available": available,
        }
        if not available:
            _fail(checks, "torch_cuda_available", detail)
            return
        if v.startswith("2.5.1") and cuda_v == "12.1":
            _ok(checks, "torch", detail)
        else:
            # 关键：torch 版本不兼容必须 fail
            _fail(checks, "torch", {**detail, "expected": "2.5.1+cu121"})
    except ImportError:
        _fail(checks, "torch", {"error": "torch not installed"})


def check_gpu(checks):
    try:
        import torch
        if not torch.cuda.is_available():
            _fail(checks, "gpu", {"error": "CUDA not available"})
            return
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        detail = {"name": name, "compute_capability": f"{cap[0]}.{cap[1]}"}
        # L40 的 compute capability 是 8.9 (sm89)
        if "L40" in name:
            _ok(checks, "gpu", detail)
        else:
            _warn(checks, "gpu", {**detail, "expected": "L40"})
        # 查询显存
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            detail["total_memory_mb"] = total // (1024 * 1024)
            checks[-1]["detail"] = detail
        except Exception:
            pass
    except Exception as e:
        _fail(checks, "gpu", {"error": str(e)})


def check_packages(checks):
    # 关键包：版本不匹配必须 fail
    # 非关键包：版本不匹配只 warn
    critical_pkgs = {
        "transformers": "4.46.3",
        "vllm": "0.6.6.post1",
        "lm_eval": "0.4.5",
        "peft": "0.13.2",
    }
    non_critical_pkgs = {
        "accelerate": "1.1.1",
        "datasets": "2.20.0",
        "safetensors": None,
    }
    for pkg, expected in critical_pkgs.items():
        v = _version(pkg)
        if v is None:
            _fail(checks, f"pkg_{pkg}", {"error": "not installed"})
        elif expected and v != expected:
            _fail(checks, f"pkg_{pkg}", {"version": v, "expected": expected})
        else:
            _ok(checks, f"pkg_{pkg}", {"version": v})
    for pkg, expected in non_critical_pkgs.items():
        v = _version(pkg)
        if v is None:
            _warn(checks, f"pkg_{pkg}", {"error": "not installed"})
        elif expected and v != expected:
            _warn(checks, f"pkg_{pkg}", {"version": v, "expected": expected})
        else:
            _ok(checks, f"pkg_{pkg}", {"version": v})


def check_merged_model(checks):
    """检查 merged model 的 safetensors 分片完整性。"""
    for label, mdir in MERGED_DIRS:
        name = f"merged_model_{label}"
        if not mdir.exists():
            _warn(checks, name, {
                "path": str(mdir),
                "status": "not_found (merge_lora.py 尚未执行)"
            })
            continue
        # 检查 config.json
        config = mdir / "config.json"
        if not config.is_file():
            _fail(checks, name, {"path": str(mdir), "error": "config.json 缺失"})
            continue
        # 检查 safetensors
        index = mdir / "model.safetensors.index.json"
        if index.is_file():
            try:
                idx = json.loads(index.read_text(encoding="utf-8"))
                weight_map = idx.get("weight_map", {})
                shards = sorted(set(weight_map.values()))
                missing = [s for s in shards if not (mdir / s).is_file()]
                zero_size = [s for s in shards if (mdir / s).is_file()
                             and (mdir / s).stat().st_size == 0]
                if missing or zero_size:
                    _fail(checks, name, {
                        "path": str(mdir),
                        "shards": shards,
                        "missing": missing,
                        "zero_size": zero_size,
                    })
                else:
                    _ok(checks, name, {
                        "path": str(mdir),
                        "num_shards": len(shards),
                        "shards": shards,
                    })
            except Exception as e:
                _fail(checks, name, {"path": str(mdir), "error": str(e)})
        else:
            # 单文件模式
            sf = mdir / "model.safetensors"
            if sf.is_file() and sf.stat().st_size > 0:
                _ok(checks, name, {"path": str(mdir), "single_file": True})
            else:
                _fail(checks, name, {
                    "path": str(mdir),
                    "error": "model.safetensors 缺失或为空"
                })


def check_local_tasks(checks):
    """检查三个本地 task 的 YAML 文件存在且语法可解析。"""
    import glob as glb
    try:
        import yaml
    except ImportError:
        yaml = None

    # 自定义 YAML loader 处理 !function 标签
    if yaml:
        class _FunctionLoader(yaml.SafeLoader):
            pass
        def _construct_function(loader, node):
            if isinstance(node, yaml.ScalarNode):
                return loader.construct_scalar(node)
            return str(node.value)
        _FunctionLoader.add_constructor("!function", _construct_function)

    for task in REQUIRED_TASKS:
        name = f"local_task_{task}"
        yamls = glb.glob(str(EVAL_TASKS_DIR / "*.yaml"))
        found = False
        for yp in yamls:
            try:
                content = Path(yp).read_text(encoding="utf-8")
                if yaml:
                    data = yaml.load(content, Loader=_FunctionLoader)
                    if data and data.get("task") == task:
                        found = True
                        _ok(checks, name, {"yaml": yp, "task": task})
                        break
                else:
                    if f"task: {task}" in content:
                        found = True
                        _warn(checks, name, {
                            "yaml": yp,
                            "note": "yaml module not installed, only text-checked"
                        })
                        break
            except Exception as e:
                _fail(checks, name, {"yaml": yp, "error": str(e)})
                continue
        if not found:
            _fail(checks, name, {"error": f"task '{task}' not found in eval_tasks/"})
    # 检查 math_utils.py（generation-only 模式不依赖 math_verify，缺失只 warn）
    mu = EVAL_TASKS_DIR / "math_utils.py"
    if mu.is_file():
        _ok(checks, "math_utils", {"path": str(mu)})
    else:
        _warn(checks, "math_utils", {
            "error": "eval_tasks/math_utils.py 缺失（generation-only 模式非阻塞）",
        })


def check_hf_datasets(checks):
    """尝试检测 Hugging Face 数据集可访问性（不下载，只查 metadata）。"""
    datasets_to_check = [
        ("HuggingFaceH4/MATH-500", "default"),
        ("Maxwell-Jia/AIME_2024", None),
        ("math-ai/aime25", None),
    ]
    try:
        from datasets import load_dataset_builder
    except ImportError:
        _warn(checks, "hf_datasets", {"error": "datasets 包未安装"})
        return
    for ds, config in datasets_to_check:
        name = f"hf_dataset_{ds.replace('/', '_')}"
        try:
            if config:
                load_dataset_builder(ds, config)
            else:
                load_dataset_builder(ds)
            _ok(checks, name, {"dataset": ds})
        except Exception as e:
            _warn(checks, name, {"dataset": ds, "error": str(e)[:200]})


def check_disk_space(checks):
    """检查磁盘剩余空间。"""
    try:
        total, used, free = shutil.disk_usage(str(PROJECT_DIR))
        free_gb = free / (1024 ** 3)
        detail = {
            "total_gb": round(total / (1024 ** 3), 1),
            "used_gb": round(used / (1024 ** 3), 1),
            "free_gb": round(free_gb, 1),
        }
        if free_gb < 10:
            _warn(checks, "disk_space", {**detail, "note": "剩余 < 10GB，可能不足"})
        else:
            _ok(checks, "disk_space", detail)
    except Exception as e:
        _warn(checks, "disk_space", {"error": str(e)})


def check_cpu_memory(checks):
    """检查 CPU 内存。"""
    try:
        import psutil
        vm = psutil.virtual_memory()
        detail = {
            "total_gb": round(vm.total / (1024 ** 3), 1),
            "available_gb": round(vm.available / (1024 ** 3), 1),
            "percent_used": vm.percent,
        }
        _ok(checks, "cpu_memory", detail)
    except ImportError:
        # Linux /proc/meminfo
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                info = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        info[parts[0].strip()] = parts[1].strip()
            total_kb = int(info.get("MemTotal", "0 kB").split()[0])
            avail_kb = int(info.get("MemAvailable", "0 kB").split()[0])
            _ok(checks, "cpu_memory", {
                "total_gb": round(total_kb / (1024 ** 2), 1),
                "available_gb": round(avail_kb / (1024 ** 2), 1),
                "source": "/proc/meminfo",
            })
        except Exception:
            _warn(checks, "cpu_memory", {"error": "无法获取内存信息"})
    except Exception as e:
        _warn(checks, "cpu_memory", {"error": str(e)})


def check_gpu_free(checks):
    """检查目标 GPU 是否空闲。"""
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if "," in gpu_id:
        _fail(checks, "gpu_selection", {
            "cuda_visible_devices": gpu_id,
            "error": "包含逗号，只允许单卡",
        })
        return
    _ok(checks, "gpu_selection", {"cuda_visible_devices": gpu_id})
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_id}",
             "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode().strip()
        if out:
            _warn(checks, "gpu_free", {
                "gpu_index": gpu_id,
                "processes": out,
                "note": "GPU 上有进程，评测前需确保是空闲的",
            })
        else:
            _ok(checks, "gpu_free", {"gpu_index": gpu_id, "status": "idle"})
    except FileNotFoundError:
        _warn(checks, "gpu_free", {"error": "nvidia-smi 不可用（可能在非 GPU 机器上运行）"})
    except Exception as e:
        _warn(checks, "gpu_free", {"error": str(e)})


def check_pip_conflicts(checks):
    """运行 pip check 检测依赖冲突。pip check 返回非零是关键问题。"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            _ok(checks, "pip_check", {"status": "no conflicts"})
        else:
            # 关键：pip check 冲突必须 fail
            _fail(checks, "pip_check", {
                "status": "conflicts detected",
                "output": result.stdout[:500],
            })
    except Exception as e:
        _warn(checks, "pip_check", {"error": str(e)})


def check_chunked_prefill_config(checks):
    """静态检查 run_eval_single_l40_vllm.sh 中的 vLLM 调度关键参数。

    检查项:
      * enable_chunked_prefill=True 出现在 model_args 中
      * MAX_MODEL_LEN >= 32768
      * MAX_GEN_TOKS == 32768
      * tensor_parallel_size=1 出现在 model_args 中
    任一检查失败均使用 _fail()。
    """
    import re

    sh_path = PROJECT_DIR / "scripts" / "run_eval_single_l40_vllm.sh"
    name = "chunked_prefill_config"
    if not sh_path.is_file():
        _fail(checks, name, {"error": f"{sh_path} 不存在"})
        return

    content = sh_path.read_text(encoding="utf-8")
    errors = []

    # 1. 检查 enable_chunked_prefill=True
    if "enable_chunked_prefill=True" not in content:
        errors.append("enable_chunked_prefill=True 未在 model_args 中找到")

    # 2. 检查 MAX_MODEL_LEN >= 32768
    max_model_len = None
    # 匹配 MAX_MODEL_LEN="${MAX_MODEL_LEN:-NUMBER}" 形式（带默认值）
    m = re.search(r'MAX_MODEL_LEN="\$\{MAX_MODEL_LEN:-(\d+)\}"', content)
    if m:
        max_model_len = int(m.group(1))
    else:
        # 匹配直接赋值 MAX_MODEL_LEN=NUMBER
        m = re.search(r'^\s*MAX_MODEL_LEN=(\d+)', content, re.MULTILINE)
        if m:
            max_model_len = int(m.group(1))
    if max_model_len is None:
        errors.append("无法从脚本中解析 MAX_MODEL_LEN")
    elif max_model_len < 32768:
        errors.append(f"MAX_MODEL_LEN={max_model_len} < 32768")

    # 3. 检查 MAX_GEN_TOKS == 32768
    max_gen_toks = None
    m = re.search(r'^\s*MAX_GEN_TOKS=(\d+)', content, re.MULTILINE)
    if m:
        max_gen_toks = int(m.group(1))
    if max_gen_toks is None:
        errors.append("无法从脚本中解析 MAX_GEN_TOKS")
    elif max_gen_toks != 32768:
        errors.append(f"MAX_GEN_TOKS={max_gen_toks} != 32768")

    # 4. 检查 tensor_parallel_size=1
    if "tensor_parallel_size=1" not in content:
        errors.append("tensor_parallel_size=1 未在 model_args 中找到")

    if errors:
        _fail(checks, name, {
            "script": str(sh_path),
            "errors": errors,
            "max_model_len": max_model_len,
            "max_gen_toks": max_gen_toks,
        })
    else:
        _ok(checks, name, {
            "script": str(sh_path),
            "max_model_len": max_model_len,
            "max_gen_toks": max_gen_toks,
            "enable_chunked_prefill": True,
            "tensor_parallel_size": 1,
        })


def check_vllm_scheduler_config(checks):
    """构造 vllm.config.SchedulerConfig 验证参数兼容性。

    使用与 run_eval_single_l40_vllm.sh level-1 fallback 一致的参数构造
    SchedulerConfig，验证 max_num_batched_tokens / max_num_seqs /
    max_model_len / enable_chunked_prefill 之间不存在约束冲突。
    """
    name = "vllm_scheduler_config"
    params = {
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 32,
        "max_model_len": 40960,
        "enable_chunked_prefill": True,
    }
    try:
        from vllm.config import SchedulerConfig
    except ImportError:
        _warn(checks, name, {"error": "vllm 未安装，无法验证 SchedulerConfig"})
        return
    try:
        SchedulerConfig(
            max_num_batched_tokens=params["max_num_batched_tokens"],
            max_num_seqs=params["max_num_seqs"],
            max_model_len=params["max_model_len"],
            enable_chunked_prefill=params["enable_chunked_prefill"],
        )
        _ok(checks, name, params)
    except Exception as e:
        _fail(checks, name, {
            "error": str(e),
            "params": params,
        })


def main():
    checks = []

    check_python(checks)
    check_torch(checks)
    check_gpu(checks)
    check_packages(checks)
    check_merged_model(checks)
    check_local_tasks(checks)
    check_hf_datasets(checks)
    check_disk_space(checks)
    check_cpu_memory(checks)
    check_gpu_free(checks)
    check_pip_conflicts(checks)
    check_chunked_prefill_config(checks)
    check_vllm_scheduler_config(checks)

    # 汇总
    n_ok = sum(1 for c in checks if c["status"] == "ok")
    n_warn = sum(1 for c in checks if c["status"] == "warn")
    n_fail = sum(1 for c in checks if c["status"] == "fail")

    report = {
        "project_dir": str(PROJECT_DIR),
        "platform": platform.platform(),
        "checks": checks,
        "summary": {
            "ok": n_ok,
            "warn": n_warn,
            "fail": n_fail,
            "critical_fail": n_fail,
        },
        "overall_status": "pass" if n_fail == 0 else "fail",
    }

    # 写入 results/environment_diagnostic.json
    out_dir = PROJECT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "environment_diagnostic.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, out_path)

    # 打印摘要
    print(f"环境诊断完成: {n_ok} ok, {n_warn} warn, {n_fail} fail")
    print(f"报告: {out_path}")
    print()
    for c in checks:
        status_icon = {"ok": "[OK]", "warn": "[WARN]", "fail": "[FAIL]"}[c["status"]]
        print(f"  {status_icon} {c['name']}: {json.dumps(c.get('detail', {}), ensure_ascii=False)[:120]}")

    # 任何 fail 返回非零
    if n_fail > 0:
        print(f"\n[ERROR] {n_fail} 项关键检查失败，请修复后再评测。", file=sys.stderr)
        sys.exit(1)
    print("\n[OK] 环境诊断通过（警告项请酌情关注）。")
    sys.exit(0)


if __name__ == "__main__":
    main()
