#!/usr/bin/env python3
"""Read-only inspector for the OpenEW-SA dataset (G0 data audit).

This script ONLY reads and reports. It makes **no assumptions** about dataset
semantics, imposes **no attribute taxonomy**, and never writes back to the data
directory. Its job is to answer "what is actually here?" so the G0 audit
checklist (docs/G0_data_audit_plan.md) can be filled from real evidence.

Usage:
    python src/audit_openew_sa.py --root <DATASET_PATH> --out runs/g0_audit/

Output:
    <out>/audit_report.json   machine-readable inventory
    stdout                    human-readable summary

Optional dependencies (numpy, pandas, pyarrow, h5py, PyYAML) are used if present
and degraded gracefully if absent — their absence is reported, never fatal.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

# ---- optional deps, guarded ------------------------------------------------
_OPT = {}
for _m in ("numpy", "pandas", "h5py", "yaml"):
    try:
        _OPT[_m] = __import__(_m)
    except Exception:  # pragma: no cover - environment dependent
        _OPT[_m] = None

MAX_UNIQUE = 50          # report full value set only below this cardinality
SAMPLE_ROWS = 5          # preview rows for tabular files
HASH_BYTES = 1 << 20     # hash only first 1 MiB for speed (integrity fingerprint)


def _sha1_prefix(path: Path) -> str:
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            h.update(f.read(HASH_BYTES))
    except Exception as e:
        return f"<unreadable: {e}>"
    return h.hexdigest()


def _describe_tabular(path: Path) -> dict:
    """Columns, dtypes, low-cardinality unique values, numeric stats. No semantics."""
    pd = _OPT["pandas"]
    if pd is None:
        return {"note": "pandas not installed; cannot parse tabular file"}
    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, nrows=200_000)
    except Exception as e:
        return {"error": f"failed to read: {e}"}

    cols = {}
    for c in df.columns:
        s = df[c]
        info = {"dtype": str(s.dtype), "n_missing": int(s.isna().sum())}
        nun = int(s.nunique(dropna=True))
        info["n_unique"] = nun
        if nun <= MAX_UNIQUE:
            # candidate categorical / label field — list its actual values verbatim
            vc = s.value_counts(dropna=True)
            info["values"] = {str(k): int(v) for k, v in vc.items()}
        elif str(s.dtype).startswith(("int", "float")):
            info["stats"] = {
                "min": float(s.min()), "max": float(s.max()),
                "mean": float(s.mean()), "std": float(s.std()),
            }
        cols[str(c)] = info
    return {
        "n_rows_read": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": cols,
        "sample": df.head(SAMPLE_ROWS).to_dict(orient="records"),
    }


def _describe_npy(path: Path) -> dict:
    np = _OPT["numpy"]
    if np is None:
        return {"note": "numpy not installed; cannot parse array file"}
    try:
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as z:
                return {"arrays": {k: {"shape": list(z[k].shape),
                                       "dtype": str(z[k].dtype)} for k in z.files}}
        arr = np.load(path, allow_pickle=False, mmap_mode="r")
        return {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    except Exception as e:
        return {"error": f"failed to read: {e}"}


def _describe_hdf5(path: Path) -> dict:
    h5py = _OPT["h5py"]
    if h5py is None:
        return {"note": "h5py not installed; cannot parse HDF5 file"}
    out = {}
    try:
        with h5py.File(path, "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    out[name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
            f.visititems(visit)
            out["_attrs"] = {k: str(v) for k, v in f.attrs.items()}
    except Exception as e:
        return {"error": f"failed to read: {e}"}
    return out


def _describe_structured_text(path: Path) -> dict:
    try:
        text = path.read_text(errors="replace")[: 1 << 16]
    except Exception as e:
        return {"error": f"failed to read: {e}"}
    if path.suffix.lower() in (".json",):
        try:
            obj = json.loads(text)
            return {"top_level_type": type(obj).__name__,
                    "keys": list(obj.keys())[:MAX_UNIQUE] if isinstance(obj, dict) else None,
                    "len": len(obj) if hasattr(obj, "__len__") else None}
        except Exception as e:
            return {"note": f"json preview only (truncated/invalid?): {e}"}
    if path.suffix.lower() in (".yaml", ".yml") and _OPT["yaml"] is not None:
        try:
            obj = _OPT["yaml"].safe_load(text)
            return {"top_level_type": type(obj).__name__,
                    "keys": list(obj.keys())[:MAX_UNIQUE] if isinstance(obj, dict) else None}
        except Exception as e:
            return {"note": f"yaml preview only: {e}"}
    return {"preview_chars": len(text)}


DISPATCH = {
    ".csv": _describe_tabular, ".tsv": _describe_tabular, ".parquet": _describe_tabular,
    ".npy": _describe_npy, ".npz": _describe_npy,
    ".h5": _describe_hdf5, ".hdf5": _describe_hdf5,
    ".json": _describe_structured_text, ".yaml": _describe_structured_text,
    ".yml": _describe_structured_text,
}


def audit(root: Path) -> dict:
    files = [p for p in root.rglob("*") if p.is_file()]
    by_ext = collections.Counter(p.suffix.lower() for p in files)
    total_bytes = sum((p.stat().st_size for p in files), 0)

    described = []
    for p in sorted(files):
        rec = {
            "path": str(p.relative_to(root)),
            "ext": p.suffix.lower(),
            "size_bytes": p.stat().st_size,
            "sha1_1mib": _sha1_prefix(p),
        }
        fn = DISPATCH.get(p.suffix.lower())
        if fn is not None:
            rec["inspect"] = fn(p)
        described.append(rec)

    return {
        "root": str(root),
        "optional_deps_present": {k: (v is not None) for k, v in _OPT.items()},
        "summary": {
            "n_files": len(files),
            "total_bytes": total_bytes,
            "by_extension": dict(by_ext),
        },
        "files": described,
        "DISCLAIMER": (
            "Read-only inventory. No dataset semantics assumed; no attribute "
            "taxonomy imposed. Fill docs/G0_data_audit_plan.md from this evidence."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="path to OpenEW-SA dataset (read-only)")
    ap.add_argument("--out", default="runs/g0_audit", help="output dir for audit_report.json")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: dataset path does not exist: {root}", file=sys.stderr)
        return 2

    report = audit(root)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2, default=str))

    s = report["summary"]
    print(f"Audited {root}")
    print(f"  files: {s['n_files']}   total: {s['total_bytes'] / 1e6:.1f} MB")
    print(f"  by extension: {s['by_extension']}")
    missing = [k for k, v in report["optional_deps_present"].items() if not v]
    if missing:
        print(f"  NOTE: optional deps absent (some files not parsed): {missing}")
    print(f"  wrote: {out_dir / 'audit_report.json'}")
    print("  -> next: fill docs/G0_data_audit_plan.md from the report; do NOT assume semantics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
