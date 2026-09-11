"""Pre-flight: every note's front matter must parse as YAML before it is pushed.

A colon inside an unquoted value breaks the site build, and the failure surfaces
only as a failed deploy after the merge. Run this before pushing docs changes.

Run: uv run --script scripts/check-front-matter.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is a Jekyll-side concern
    print("PyYAML is not installed; skipping (uv pip install pyyaml)")
    sys.exit(0)

failures = []
checked = 0
for path in sorted((ROOT / "docs").rglob("*.md")):
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        continue
    end = text.find("\n---", 4)
    if end == -1:
        failures.append((path, "front matter is not closed"))
        continue
    checked += 1
    try:
        yaml.safe_load(text[4:end])
    except yaml.YAMLError as error:
        failures.append((path, str(error).splitlines()[0]))

print(f"checked {checked} documents with front matter")
for path, reason in failures:
    print(f"FAIL {path.relative_to(ROOT)}: {reason}")
sys.exit(1 if failures else 0)
