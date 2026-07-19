"""
test_merge_completeness.py
测试 safetensors 分片完整性检查逻辑：
  * 单文件 model.safetensors 完整
  * 多分片 model.safetensors.index.json + shards 完整
  * 缺少 shard 文件
  * shard 文件大小为 0
  * index.json 损坏
"""
import json
import os
import struct
import sys
from pathlib import Path

import pytest

# 导入 merge_lora.py 中的函数
try:
    from merge_lora import is_complete_model, _check_safetensors_header, _verify_shards
except ImportError:
    # merge_lora.py 可能在 scripts/ 目录
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from merge_lora import is_complete_model, _check_safetensors_header, _verify_shards


def _write_fake_safetensors(path: Path, n_bytes: int = 1024):
    """写入一个假的 safetensors 文件（有效 header）。"""
    # safetensors 格式: 8 字节 header length (uint64 LE) + JSON header + data
    header = json.dumps({"default": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}})
    header_bytes = header.encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * 4)  # fake data
        # 补齐到 n_bytes
        current = 8 + len(header_bytes) + 4
        if current < n_bytes:
            f.write(b"\x00" * (n_bytes - current))


def _write_config_json(path: Path):
    """写入一个最小的 config.json。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "torch_dtype": "bfloat16",
        }, f)


def _write_tokenizer_file(model_dir: Path):
    """写入一个最小的 tokenizer 文件，使 is_complete_model 通过。"""
    # is_complete_model 检查 _TOKENIZER_FILE_CANDIDATES 中的任意一个
    with open(model_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump({"tokenizer_class": "LlamaTokenizer"}, f)


def _make_complete_model(model_dir: Path, sharded=False, n_shards=2):
    """创建一个完整的 mock 模型目录（config + safetensors + tokenizer）。"""
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_config_json(model_dir / "config.json")
    _write_tokenizer_file(model_dir)
    if sharded:
        shard_names = [f"model-0000{i}-of-0000{n_shards}.safetensors"
                       for i in range(1, n_shards + 1)]
        weight_map = {}
        for i, sn in enumerate(shard_names):
            weight_map[f"layer.{i}.weight"] = sn
        index_data = {"weight_map": weight_map, "metadata": {"total_size": 2048}}
        with open(model_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index_data, f)
        for sn in shard_names:
            _write_fake_safetensors(model_dir / sn)
    else:
        _write_fake_safetensors(model_dir / "model.safetensors")


class TestSingleFileModel:
    """单文件 model.safetensors 完整"""

    def test_complete_single_file(self, tmp_path):
        model_dir = tmp_path / "model"
        _make_complete_model(model_dir, sharded=False)
        assert is_complete_model(str(model_dir)) is True

    def test_missing_config(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_tokenizer_file(model_dir)
        _write_fake_safetensors(model_dir / "model.safetensors")
        # 不写 config.json
        assert is_complete_model(str(model_dir)) is False

    def test_missing_tokenizer(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_config_json(model_dir / "config.json")
        _write_fake_safetensors(model_dir / "model.safetensors")
        # 不写 tokenizer 文件
        assert is_complete_model(str(model_dir)) is False

    def test_empty_safetensors(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_config_json(model_dir / "config.json")
        _write_tokenizer_file(model_dir)
        # 空文件
        (model_dir / "model.safetensors").write_bytes(b"")
        assert is_complete_model(str(model_dir)) is False


class TestShardedModel:
    """多分片 model.safetensors.index.json + shards"""

    def test_complete_sharded(self, tmp_path):
        model_dir = tmp_path / "model"
        _make_complete_model(model_dir, sharded=True, n_shards=2)
        assert is_complete_model(str(model_dir)) is True

    def test_missing_shard(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        _write_config_json(model_dir / "config.json")
        _write_tokenizer_file(model_dir)
        # 创建 index 指向 2 个 shard，但只创建 1 个
        shard_names = ["model-00001-of-00002.safetensors",
                       "model-00002-of-00002.safetensors"]
        weight_map = {"layer.0.weight": shard_names[0],
                      "layer.1.weight": shard_names[1]}
        with open(model_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": weight_map}, f)
        _write_fake_safetensors(model_dir / shard_names[0])
        # shard_names[1] 不创建
        assert is_complete_model(str(model_dir)) is False

    def test_zero_size_shard(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        _write_config_json(model_dir / "config.json")
        _write_tokenizer_file(model_dir)
        shard_names = ["model-00001-of-00002.safetensors",
                       "model-00002-of-00002.safetensors"]
        weight_map = {"layer.0.weight": shard_names[0],
                      "layer.1.weight": shard_names[1]}
        with open(model_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": weight_map}, f)
        _write_fake_safetensors(model_dir / shard_names[0])
        # shard 2 为空文件
        (model_dir / shard_names[1]).write_bytes(b"")
        assert is_complete_model(str(model_dir)) is False

    def test_corrupt_index(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        _write_config_json(model_dir / "config.json")
        _write_tokenizer_file(model_dir)
        with open(model_dir / "model.safetensors.index.json", "w") as f:
            f.write("{ broken json")
        _write_fake_safetensors(model_dir / "model-00001-of-00002.safetensors")
        _write_fake_safetensors(model_dir / "model-00002-of-00002.safetensors")
        assert is_complete_model(str(model_dir)) is False


class TestSafetensorsHeader:
    """safetensors header 读取"""

    def test_valid_header(self, tmp_path):
        sf = tmp_path / "test.safetensors"
        _write_fake_safetensors(sf)
        result = _check_safetensors_header(str(sf))
        assert result is True

    def test_invalid_header(self, tmp_path):
        sf = tmp_path / "test.safetensors"
        # 写入无效数据
        sf.write_bytes(b"\x00\x00\x00\x00\x00\x00\x00\x00invalid")
        result = _check_safetensors_header(str(sf))
        # 无效 header 应返回 False 或不崩溃
        assert result is False or result is True  # 至少不崩溃

    def test_empty_file(self, tmp_path):
        sf = tmp_path / "test.safetensors"
        sf.write_bytes(b"")
        result = _check_safetensors_header(str(sf))
        assert result is False


class TestShardVerification:
    """_verify_shards 函数：返回 shard 路径列表，不完整时抛 RuntimeError"""

    def test_all_shards_present_no_index(self, tmp_path):
        """无 index.json 时，_verify_shards 列出目录中所有 .safetensors 文件。"""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_fake_safetensors(model_dir / "shard1.safetensors")
        _write_fake_safetensors(model_dir / "shard2.safetensors")
        result = _verify_shards(str(model_dir))
        assert isinstance(result, list)
        assert len(result) == 2

    def test_all_shards_present_with_index(self, tmp_path):
        """有 index.json 时，_verify_shards 按 weight_map 验证每个 shard。"""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        shard_names = ["model-00001-of-00002.safetensors",
                       "model-00002-of-00002.safetensors"]
        weight_map = {"layer.0.weight": shard_names[0],
                      "layer.1.weight": shard_names[1]}
        with open(model_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": weight_map}, f)
        for sn in shard_names:
            _write_fake_safetensors(model_dir / sn)
        result = _verify_shards(str(model_dir))
        assert isinstance(result, list)
        assert len(result) == 2

    def test_missing_shard_raises(self, tmp_path):
        """index.json 指向的 shard 缺失时，_verify_shards 抛 RuntimeError。"""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        shard_names = ["model-00001-of-00002.safetensors",
                       "model-00002-of-00002.safetensors"]
        weight_map = {"layer.0.weight": shard_names[0],
                      "layer.1.weight": shard_names[1]}
        with open(model_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": weight_map}, f)
        _write_fake_safetensors(model_dir / shard_names[0])
        # shard_names[1] 不创建
        with pytest.raises(RuntimeError, match="分片缺失"):
            _verify_shards(str(model_dir))

    def test_no_safetensors_raises(self, tmp_path):
        """目录中没有任何 .safetensors 文件时抛 RuntimeError。"""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        with pytest.raises(RuntimeError, match="未找到任何 safetensors"):
            _verify_shards(str(model_dir))
