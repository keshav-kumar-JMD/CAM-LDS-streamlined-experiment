"""Run Notebook 10 round-two repairs without modifying analysis/ or report/."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def digest_tree(path: Path):
    result = {}
    if not path.exists():
        return result
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        h = hashlib.sha256()
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
        result[str(file.relative_to(path))] = h.hexdigest()
    return result


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "analysis_round2"
    before = {"analysis": digest_tree(root / "analysis"), "report": digest_tree(root / "report")}
    sys.path.insert(0, str(root))
    from analysis_round2.code.figure_fixes import regenerate_notebook10_figures
    result = regenerate_notebook10_figures(root, output_root=output)
    after = {"analysis": digest_tree(root / "analysis"), "report": digest_tree(root / "report")}
    changed = {
        section: sorted(set(before[section]) | set(after[section]))
        for section in before
        if before[section] != after[section]
    }
    audit = {
        "renderer_result": result,
        "existing_analysis_unchanged": before["analysis"] == after["analysis"],
        "report_unchanged": before["report"] == after["report"],
        "changed_baseline_sections": changed,
    }
    (output / "manifests").mkdir(parents=True, exist_ok=True)
    (output / "manifests" / "isolation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if changed:
        raise RuntimeError(f"Isolation failure: existing paths changed: {changed}")


if __name__ == "__main__":
    main()
