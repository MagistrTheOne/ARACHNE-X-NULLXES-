"""
LoRA key format and state roundtrip without importing arachne_x package (__init__ pulls loader/triton).
Run: python -m unittest tests.test_lora_init_roundtrip -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent


def _load_lora_utils():
    path = ROOT / "arachne_x" / "modules" / "lora_utils.py"
    spec = importlib.util.spec_from_file_location("_lora_utils_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _B(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d)


class _Toy(nn.Module):
    def __init__(self, d: int = 16):
        super().__init__()
        self.x_embedder = nn.Identity()
        self.blocks = nn.ModuleList([_B(d)])


class TestLoraInitRoundtrip(unittest.TestCase):
    def test_prefix_roundtrip(self):
        lu = _load_lora_utils()
        name = "blocks.0.lin"
        prefix = lu.module_path_to_lora_prefix(name)
        back = prefix.replace("lora___lorahyphen___", "").replace("___lorahyphen___", ".")
        self.assertEqual(back, name)

    def test_build_and_reload_state(self):
        lu = _load_lora_utils()
        from safetensors.torch import load_file, save_file

        m1 = _Toy(16)
        st = lu.build_initial_lora_state_dict(
            m1,
            rank=4,
            alpha=8.0,
            name_filter=lambda n, m: lu.default_avatar_train_lora_filter(n, m),
        )
        net = lu.create_lora_network(m1, st, 1.0, 4, 8.0)
        inc = net.load_state_dict(st, strict=True)
        self.assertFalse(inc.missing_keys)
        self.assertFalse(inc.unexpected_keys)

        fd, path = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        try:
            save_file({k: v.cpu() for k, v in net.state_dict().items()}, path)
            loaded = load_file(path, device="cpu")
            m2 = _Toy(16)
            net2 = lu.create_lora_network(m2, loaded, 1.0, 4, 8.0)
            net2.load_state_dict(loaded, strict=False)
            k = next(iter(st))
            self.assertTrue(torch.allclose(net2.state_dict()[k].float(), st[k].float()))
        finally:
            if os.path.isfile(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
