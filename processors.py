"""
Processor registry: maps processor names to callables.
Each processor takes a PyArrow Table + list of input column names,
and returns a PyArrow Array (the derived column values).
"""
import hashlib
from difflib import SequenceMatcher

import lance
import pyarrow as pa
import numpy as np

REGISTRY: dict[str, callable] = {}


def register(name: str):
    def decorator(fn):
        REGISTRY[name] = fn
        return fn
    return decorator


@register("clean_name")
def clean_name(table: pa.Table, input_cols: list[str], lookup=None) -> pa.Array:
    """Normalize names: strip whitespace, title case."""
    col = table.column(input_cols[0])
    cleaned = []
    for val in col.to_pylist():
        if val is None:
            cleaned.append(None)
        else:
            cleaned.append(val.strip().title())
    return pa.array(cleaned, type=pa.string())


@register("clean_skills")
def clean_skills(table: pa.Table, input_cols: list[str], lookup=None) -> pa.Array:
    """Normalize skills: lowercase, strip, deduplicate, sort, rejoin."""
    col = table.column(input_cols[0])
    cleaned = []
    for val in col.to_pylist():
        if val is None:
            cleaned.append(None)
        else:
            skills = [s.strip().lower() for s in val.split(",")]
            skills = sorted(set(skills))
            cleaned.append(", ".join(skills))
    return pa.array(cleaned, type=pa.string())


@register("mock_embedding")
def mock_embedding(table: pa.Table, input_cols: list[str], lookup=None) -> pa.Array:
    """
    Generate a deterministic mock embedding from input columns.
    In production this would call OpenAI/a GPU model.
    Returns a fixed-length float32 list (dim=64 for demo).
    """
    dim = 64
    embeddings = []
    for i in range(len(table)):
        combined = ""
        for col_name in input_cols:
            val = table.column(col_name)[i].as_py()
            combined += str(val or "")
        # Deterministic: hash the input to seed the RNG
        seed = int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)  # unit normalize
        embeddings.append(vec.tolist())
    return pa.array(embeddings, type=pa.list_(pa.float32()))


@register("match_linkedin_profile")
def match_linkedin_profile(table: pa.Table, input_cols: list[str], lookup=None) -> pa.Array:
    """
    Cross-dataset lookup: match github users to linkedin profiles by name similarity.
    Uses the lookup config to open the linkedin Lance dataset and find best matches.
    """
    if lookup is None:
        return pa.array([None] * len(table), type=pa.string())

    # Open the external linkedin dataset
    try:
        ext_ds = lance.dataset(lookup.dataset_path)
        ext_table = ext_ds.to_table(columns=lookup.match_columns)
    except Exception:
        return pa.array([None] * len(table), type=pa.string())

    # Build name -> member_id lookup from linkedin data
    name_col = "cleaned_name"
    id_col = "member_id"
    if name_col not in ext_table.schema.names or id_col not in ext_table.schema.names:
        return pa.array([None] * len(table), type=pa.string())

    ext_names = ext_table.column(name_col).to_pylist()
    ext_ids = ext_table.column(id_col).to_pylist()
    linkedin_lookup = {}
    for name, mid in zip(ext_names, ext_ids):
        if name is not None:
            linkedin_lookup[name.lower()] = mid

    # Match each github user's cleaned_name against linkedin names
    source_col = table.column(input_cols[0])
    matches = []
    threshold = 0.8
    for val in source_col.to_pylist():
        if val is None:
            matches.append(None)
            continue
        val_lower = val.lower()
        # Exact match first
        if val_lower in linkedin_lookup:
            matches.append(linkedin_lookup[val_lower])
            continue
        # Fuzzy match
        best_score = 0.0
        best_id = None
        for ln_name, ln_id in linkedin_lookup.items():
            score = SequenceMatcher(None, val_lower, ln_name).ratio()
            if score > best_score:
                best_score = score
                best_id = ln_id
        if best_score >= threshold:
            matches.append(best_id)
        else:
            matches.append(None)

    return pa.array(matches, type=pa.string())
