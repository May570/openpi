# selfcheck.py
import sys
import importlib

print("Python:", sys.version)

def check_import(name):
    try:
        mod = importlib.import_module(name)
        if getattr(mod, "__file__", None):
            print(f"[OK] {name} -> {mod.__file__}")
        else:
            # namespace 包没有 __file__，打印 __path__
            print(f"[OK] {name} -> namespace package, __path__ = {list(getattr(mod, '__path__', []))}")
        return True
    except Exception as e:
        print(f"[FAIL] {name} -> {e}")
        return False

# 顶层包
check_import("simpler_env")
check_import("openpi")
check_import("openpi_client")

# openpi_client 子模块
check_import("openpi_client.base_policy")

# LIBERO 子模块（注意结构是 libero/libero/...）
check_import("simpler_env.utils.env")
check_import("mani_skill2_real2sim")
