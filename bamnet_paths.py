#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Общие пути проекта BAMNet."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ENV_DATA_ROOT = "BAMNET_DATA_ROOT"
project_root = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = project_root / "data" / "BAMNet-data"


def get_data_root() -> Path:
    raw = os.environ.get(ENV_DATA_ROOT)
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_DATA_ROOT


def get_data_path(*parts: str) -> Path:
    return get_data_root().joinpath(*parts)


def expand_path_vars(value: str) -> str:
    return (
        value.replace(f"${{{ENV_DATA_ROOT}}}", str(get_data_root()))
        .replace(f"${ENV_DATA_ROOT}", str(get_data_root()))
    )


def expand_config_tree(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: expand_config_tree(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [expand_config_tree(item) for item in payload]
    if isinstance(payload, str):
        return os.path.expanduser(os.path.expandvars(expand_path_vars(payload)))
    return payload
