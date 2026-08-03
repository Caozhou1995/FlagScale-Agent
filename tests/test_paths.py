# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for paths.py FLAGSCALE_HOME override."""

import os
import tempfile
import pytest
from unittest.mock import patch
from pathlib import Path


def test_default_uses_home():
    """Without FLAGSCALE_HOME, uses ~/.flagscale."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("FLAGSCALE_HOME", None)
        from flagscale_agent.react.paths import get_dot_flagscale_root
        result = get_dot_flagscale_root()
        assert result == os.path.join(str(Path.home()), ".flagscale")


def test_flagscale_home_override():
    """FLAGSCALE_HOME overrides default path."""
    with tempfile.TemporaryDirectory() as tmp:
        custom_path = os.path.join(tmp, "custom_flagscale")
        with patch.dict(os.environ, {"FLAGSCALE_HOME": custom_path}):
            from flagscale_agent.react.paths import get_dot_flagscale_root
            result = get_dot_flagscale_root()
            assert result == os.path.abspath(custom_path)
            assert os.path.isdir(result)


def test_flagscale_home_relative_path():
    """Relative FLAGSCALE_HOME is converted to absolute."""
    with patch.dict(os.environ, {"FLAGSCALE_HOME": "./relative_dir"}):
        from flagscale_agent.react.paths import get_dot_flagscale_root
        result = get_dot_flagscale_root()
        assert os.path.isabs(result)
        assert result.endswith("relative_dir")


def test_flagscale_home_empty_string_uses_default():
    """Empty FLAGSCALE_HOME falls back to default."""
    with patch.dict(os.environ, {"FLAGSCALE_HOME": ""}):
        from flagscale_agent.react.paths import get_dot_flagscale_root
        result = get_dot_flagscale_root()
        assert result == os.path.join(str(Path.home()), ".flagscale")
