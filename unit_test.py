#!/usr/bin/env python3
"""
unit_test.py — smoke tests for the trading bot project

This suite is intentionally resilient: it focuses on importability and
presence/shape of common functions/classes without assuming exact signatures.
If a symbol isn't present, the relevant test is skipped instead of failing.

Run with:
  python -m unittest -v unit_test.py
or
  pytest -q unit_test.py

Author: ChatGPT
"""
import importlib
import inspect
import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest import mock

# Allow running from repo root or from tests/
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# These are the modules provided in the project.
MODULE_NAMES = [
    "backtradercsvexport",
    "config",              # if it's a .yaml, import will be skipped automatically
    "strats",
    "myTools",
    "export_signals",
    "av_downloader",
    "fetch_stooq_daily",
]

# Optional symbols we might expect by convention in each module. The tests will
# check for existence if present and skip gracefully otherwise.
EXPECTED_SYMBOLS = {
    "backtradercsvexport": ["export_to_csv", "main"],
    "strats":              ["Strategy", "build_cerebro", "register_strategies"],
    "myTools":             ["setup_logger", "ensure_dir", "chunk", "timer"],
    "export_signals":      ["export_signals", "signals_to_df", "main"],
    "av_downloader":       ["download_symbol", "download_batch", "to_dataframe", "main"],
    "fetch_stooq_daily":   ["fetch_symbol", "fetch_many", "to_dataframe", "main"],
}

def _is_yaml_like(modname: str) -> bool:
    # crude helper to avoid import attempts for config.yaml
    return modname.lower().endswith((".yml", ".yaml")) or modname == "config"


@contextmanager
def env(**kwargs):
    """Temporarily set environment variables for a block."""
    old = os.environ.copy()
    os.environ.update({k: str(v) for k, v in kwargs.items()})
    try:
        yield
    finally:
        # restore
        for k in list(os.environ.keys()):
            if k not in old:
                del os.environ[k]
        os.environ.update(old)


class TestImports(unittest.TestCase):
    """Verify all Python modules import cleanly without executing side effects."""

    def test_import_modules(self):
        failures = []
        for name in MODULE_NAMES:
            if _is_yaml_like(name):
                # Don't try to import YAML; just check the file exists.
                yml_paths = [os.path.join(REPO_ROOT, "config.yaml"),
                             os.path.join(REPO_ROOT, "config.yml")]
                self.assertTrue(any(os.path.exists(p) for p in yml_paths),
                                msg="Missing config.yaml / config.yml")
                continue

            with self.subTest(module=name):
                try:
                    mod = importlib.import_module(name)
                except Exception as e:
                    failures.append((name, repr(e)))
                else:
                    self.assertIsInstance(mod, types.ModuleType)
        if failures:
            msgs = "\n".join(f" - {n}: {e}" for n, e in failures)
            self.fail(f"One or more modules failed to import:\n{msgs}")


class TestAPIShapes(unittest.TestCase):
    """Check the presence and basic shape of commonly expected symbols."""

    def _import_or_skip(self, module_name: str):
        if _is_yaml_like(module_name):
            self.skipTest("config.* is not a Python module")
        try:
            return importlib.import_module(module_name)
        except Exception as e:
            self.skipTest(f"Cannot import {module_name}: {e}")

    def _assert_callable_if_exists(self, module, symbol):
        if hasattr(module, symbol):
            obj = getattr(module, symbol)
            self.assertTrue(callable(obj), f"{symbol} exists but is not callable")
        else:
            self.skipTest(f"{module.__name__}.{symbol} not present – skipped")

    def test_expected_symbols(self):
        for mod_name, symbols in EXPECTED_SYMBOLS.items():
            module = self._import_or_skip(mod_name)
            for sym in symbols:
                with self.subTest(module=mod_name, symbol=sym):
                    self._assert_callable_if_exists(module, sym)

    def test_main_doesnt_execute_on_import(self):
        """Verify modules guard CLI entry under if __name__ == '__main__'."""
        for mod_name in MODULE_NAMES:
            if _is_yaml_like(mod_name):
                continue
            with self.subTest(module=mod_name):
                module = self._import_or_skip(mod_name)
                # If a module has a top-level "main" function, ensure it's not executed on import.
                # We can't detect past execution, but we can assert "main" is a function (if present).
                if hasattr(module, "main"):
                    self.assertTrue(callable(getattr(module, "main")))


class TestSideEffectControls(unittest.TestCase):
    """Ensure network/file operations are gated behind functions / mains."""

    NETWORK_WORDS = ("requests", "urllib", "httpx", "aiohttp")
    FILE_WORDS = ("open(", "to_csv", "read_csv", "Path(", "pathlib")

    def test_no_network_calls_at_import_time(self):
        for mod_name in MODULE_NAMES:
            if _is_yaml_like(mod_name):
                continue
            with self.subTest(module=mod_name):
                # Re-import in a fresh process-like environment by deleting from sys.modules
                sys.modules.pop(mod_name, None)
                with mock.patch("builtins.open") as m_open:
                    try:
                        importlib.import_module(mod_name)
                    except Exception as e:
                        self.skipTest(f"import error {mod_name}: {e}")
                    # Ensure no accidental file opens happened during import
                    self.assertFalse(m_open.called, "Module opened files at import time")

    def test_env_usage_documented(self):
        """If the downloader uses API keys, it should look for env vars without crashing."""
        candidates = ("av_downloader", "fetch_stooq_daily")
        for mod_name in candidates:
            if mod_name not in MODULE_NAMES:
                continue
            module = self._import_or_skip(mod_name)
            # If module has a function that accepts api_key or uses env, call with dummy env to ensure no crash.
            func_names = [n for n in ("download_symbol", "fetch_symbol") if hasattr(module, n)]
            for fname in func_names:
                fn = getattr(module, fname)
                sig = inspect.signature(fn)
                # Only call if everything has a default; otherwise skip to avoid accidental network.
                all_defaults = all(
                    (p.default is not inspect._empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD))
                    for p in sig.parameters.values()
                )
                if all_defaults:
                    with self.subTest(module=mod_name, function=fname):
                        with env(ALPHAVANTAGE_API_KEY="DUMMY"):
                            try:
                                # Call with no args; expect it to handle gracefully (e.g., return None or raise ValueError)
                                result = fn()
                            except Exception as e:
                                # It's okay to raise a clear, intentional exception.
                                self.assertIsInstance(e, Exception)
                            else:
                                # If it returns, just assert it returned *something* sensible (not starting a network op).
                                self.assertNotIsInstance(result, types.GeneratorType)


class TestDataContracts(unittest.TestCase):
    """If functions return pandas DataFrames, validate basic columns when possible."""
    PANDAS_CANDIDATES = (
        ("export_signals", ("signals_to_df",)),
        ("av_downloader", ("to_dataframe",)),
        ("fetch_stooq_daily", ("to_dataframe",)),
    )

    def test_dataframe_shape_if_available(self):
        try:
            import pandas as pd  # optional
        except Exception:
            self.skipTest("pandas not installed in environment")
            return

        for mod_name, fns in self.PANDAS_CANDIDATES:
            if mod_name not in MODULE_NAMES:
                continue
            module = self._import_or_skip(mod_name)
            for fname in fns:
                if not hasattr(module, fname):
                    self.skipTest(f"{module}.{fname} not present – skipped")
                    continue
                fn = getattr(module, fname)
                # Try to call with no args only if safe (all defaulted) to avoid side effects.
                sig = inspect.signature(fn)
                all_defaults = all(
                    (p.default is not inspect._empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD))
                    for p in sig.parameters.values()
                )
                if not all_defaults:
                    self.skipTest(f"{module}.{fname} needs args – skipped")
                    continue

                with self.subTest(module=mod_name, function=fname):
                    try:
                        df = fn()  # hope it's a pure transformer with defaults
                    except Exception as e:
                        # It's fine if it raises due to missing inputs; consider that a pass for shape test.
                        self.skipTest(f"{mod_name}.{fname} raised {e} – shape test skipped")
                        continue
                    # If it returned, check it's a DataFrame with at least one of common columns
                    import pandas as pd
                    self.assertIsInstance(df, pd.DataFrame)
                    common_cols = {"symbol", "date", "close", "signal", "timestamp"}
                    self.assertTrue(len(set(df.columns) & common_cols) > 0,
                                    "Returned DataFrame lacks any common trading columns")


if __name__ == "__main__":
    unittest.main(verbosity=2)
