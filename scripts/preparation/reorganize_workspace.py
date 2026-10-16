"""One-time, checksum-verified migration; never edits the three source areas."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "archive/pre_reorganization"
SOURCES = ("ethnoracial_data", "ethnoracial_data_outputs", "training_data")
CODE = {
    "PopulateExcel.py": "scripts/preparation/populate_excel.py",
    "brainage_dataset_config.py": "scripts/shared/datasets.py",
    "plot_brainage_ridges.py": "scripts/plots/ridges.py",
    "Residual plotting/generate_residual_plots.py": "scripts/plots/residual_style.py",
    "analyses/training_bucket_bias/correct_chinese.py": "scripts/corrections/training_buckets.py",
    "analyses/training_bucket_bias/test_correct_chinese.py": "scripts/validation/test_training_buckets.py",
    "analyses/sample_age_correction/correct.py": "scripts/corrections/sample_age.py",
    "analyses/sample_age_correction/compare_controls.py": "scripts/plots/compare_controls.py",
    "analyses/sample_age_correction/plot_residuals.py": "scripts/plots/sample_residuals.py",
    "analyses/sample_age_correction/summarize_mae.py": "scripts/statistics/sample_mae.py",
    "analyses/sample_age_correction/test_correct.py": "scripts/validation/test_sample_age.py",
    "analyses/sample_age_correction/test_plot_residuals.py": "scripts/validation/test_sample_residuals.py",
    "analyses/brainage_gap/bootstrap.py": "scripts/statistics/bootstrap.py",
    "analyses/brainage_gap/permutation.py": "scripts/statistics/permutation.py",
    "analyses/qq_plots/generate_qq_plots.py": "scripts/plots/qq_plots.py",
    "analyses/plot_audit/audit_training_plots.py": "scripts/validation/historical_audit.py",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file()
                  and not any(part in (".git", "__pycache__", ".mplconfig", "node_modules") for part in p.parts)
                  and p.name != ".DS_Store")


def snapshot():
    inventory = ARCHIVE / "migration_inventory.json"
    if inventory.exists():
        raise RuntimeError("Migration snapshot already exists; refusing to replace history")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    rows = []
    original = [p for p in ROOT.iterdir() if p.name not in
                (*SOURCES, ".git", "archive", "results", "scripts", "__pycache__", "node_modules")]
    for entry in original:
        for p in files(entry) if entry.is_dir() else [entry]:
            if p.name == ".DS_Store":
                continue
            relative = str(p.relative_to(ROOT))
            target = ARCHIVE / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            digest = sha(p)
            assert sha(target) == digest
            active = CODE.get(relative)
            if active is None and "/outputs/" in relative:
                name, tail = relative.removeprefix("analyses/").split("/outputs/", 1)
                active = f"results/analyses/{name}/{tail}"
            if active is None and relative.startswith("Residual plotting/"):
                active = relative.replace("Residual plotting/", "results/analyses/residual_history/", 1)
            if active is None and relative.startswith("graph_outputs/"):
                active = relative.replace("graph_outputs/", "results/analyses/distribution_history/", 1)
            rows.append(dict(Original=relative, Archive=str(target.relative_to(ROOT)),
                             Active=active, SHA256=digest, Bytes=p.stat().st_size))
    source_rows = {str(p.relative_to(ROOT)): sha(p) for name in SOURCES for p in (ROOT/name).rglob("*") if p.is_file()}
    inventory.write_text(json.dumps(dict(Files=rows, Protected_Sources=source_rows), indent=2)+"\n")
    print(f"Archived {len(rows)} files; pinned {len(source_rows)} protected source files")


def relocate():
    data = json.loads((ARCHIVE/"migration_inventory.json").read_text())
    for row in data["Files"]:
        if row["Active"]:
            target = ROOT/row["Active"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise RuntimeError(f"Destination exists: {target}")
            shutil.copy2(ARCHIVE/row["Original"], target)
    # Originals are removed only after every archived byte and active copy is verified.
    for row in data["Files"]:
        assert sha(ARCHIVE/row["Original"]) == row["SHA256"]
        if row["Active"]:
            assert sha(ROOT/row["Active"]) == row["SHA256"]
    for name in ("analyses", "Residual plotting", "graph_outputs"):
        shutil.rmtree(ROOT/name)
    for name in ("PopulateExcel.py", "plot_brainage_ridges.py", "brainage_dataset_config.py", "plot.py"):
        (ROOT/name).unlink()
    for directory in ("shared", "preparation", "corrections", "plots", "statistics", "validation"):
        (ROOT/"scripts"/directory/"__init__.py").touch()
    (ROOT/"scripts/__init__.py").touch()
    print("Historical outputs relocated; sources unchanged")


if __name__ == "__main__":
    snapshot()
    relocate()
