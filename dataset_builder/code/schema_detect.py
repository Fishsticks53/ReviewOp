from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List

import pandas as pd

from mappings import TARGET_COLUMN_HINTS
from utils import token_count


@dataclass
class SchemaProfile:
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    text_columns: List[str]
    primary_text_column: str | None
    target_column: str | None
    column_types: Dict[str, str]


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].dropna()


DATE_HINT_RE = re.compile(
    r"^\s*(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}-\d{2}-\d{2}T.+)\s*$"
)


def detect_schema(frame: pd.DataFrame, *, text_column_override: str | None = None) -> SchemaProfile:
    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    datetime_columns: List[str] = []
    text_columns: List[str] = []
    column_types: Dict[str, str] = {}
    text_scores: Dict[str, float] = {}
    target_column: str | None = None

    for column in frame.columns:
        if column == "source_file":
            continue
        series = _series(frame, column)
        if series.empty:
            column_types[column] = "categorical"
            categorical_columns.append(column)
            continue

        values = [str(value).strip() for value in series if str(value).strip()]
        numeric_ratio = pd.to_numeric(series, errors="coerce").notna().mean()
        datetime_candidates = [value for value in values if DATE_HINT_RE.match(value)]
        datetime_ratio = len(datetime_candidates) / max(1, len(values))
        avg_tokens = sum(token_count(value) for value in values) / max(1, len(values))
        unique_ratio = len(set(values)) / max(1, len(values))
        cardinality = len(set(values))

        if pd.api.types.is_numeric_dtype(frame[column]) or numeric_ratio >= 0.90:
            numeric_columns.append(column)
            column_types[column] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(frame[column]) or datetime_ratio >= 0.90:
            datetime_columns.append(column)
            column_types[column] = "datetime"
        elif avg_tokens > 8 and unique_ratio > 0.50:
            text_columns.append(column)
            column_types[column] = "text"
            text_scores[column] = avg_tokens
        else:
            categorical_columns.append(column)
            column_types[column] = "categorical"

        if target_column is None and cardinality < 20 and any(hint in str(column).lower() for hint in TARGET_COLUMN_HINTS):
            target_column = column

    primary_text_column = None
    if text_column_override and text_column_override in frame.columns:
        primary_text_column = text_column_override
    elif text_scores:
        primary_text_column = max(text_scores, key=text_scores.get)

    return SchemaProfile(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        text_columns=text_columns,
        primary_text_column=primary_text_column,
        target_column=target_column,
        column_types=column_types,
    )
