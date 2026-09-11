import importlib.util
import sys
from pathlib import Path

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src"
PACKAGE_SPEC = importlib.util.spec_from_file_location(
    "miracle_ad",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
if PACKAGE_SPEC is None or PACKAGE_SPEC.loader is None:
    raise RuntimeError(f"Cannot load the local MIRACLE-AD package from {PACKAGE_ROOT}")

LOCAL_PACKAGE = importlib.util.module_from_spec(PACKAGE_SPEC)
sys.modules["miracle_ad"] = LOCAL_PACKAGE
PACKAGE_SPEC.loader.exec_module(LOCAL_PACKAGE)


def pytest_sessionstart(session):
    # Tiny smoke tensors are faster and more stable without large CPU thread pools.
    torch.set_num_threads(1)
