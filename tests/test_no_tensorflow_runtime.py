"""Guard active runtime code against TensorFlow dependencies."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_RUNTIME_DIRS = (REPO_ROOT / "msfm",)
EXCLUDED_RUNTIME_PATHS = set()
EXCLUDED_RUNTIME_DIRS = {
    REPO_ROOT / "msfm" / "apps" / "deprecated",
}
FORBIDDEN_TENSORFLOW_REFERENCES = (
    "import tensorflow",
    "tensorflow_probability",
    "tf.data",
    "tf.",
)

FORBIDDEN_TFRECORD_REFERENCES = (
    "tfrecord",
    "tfr_pattern",
    "TFRecord",
)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _active_python_files() -> list[Path]:
    files: list[Path] = []
    for runtime_dir in ACTIVE_RUNTIME_DIRS:
        for path in runtime_dir.rglob("*.py"):
            if path in EXCLUDED_RUNTIME_PATHS:
                continue
            if any(_is_relative_to(path, excluded_dir) for excluded_dir in EXCLUDED_RUNTIME_DIRS):
                continue
            files.append(path)
    return sorted(files)


def test_active_runtime_code_has_no_tensorflow_references() -> None:
    """Active runtime code must not reference TensorFlow."""

    violations: list[str] = []
    for path in _active_python_files():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            matches = [pattern for pattern in FORBIDDEN_TENSORFLOW_REFERENCES if pattern in line]
            if matches:
                relpath = path.relative_to(REPO_ROOT)
                violations.append(f"{relpath}:{line_number}: {', '.join(matches)}: {line.strip()}")

    assert not violations, "Active runtime TensorFlow references found:\n" + "\n".join(violations)


def test_active_runtime_code_has_no_tfrecord_references() -> None:
    """Active runtime code must not expose TFRecord loading or generation paths."""

    violations: list[str] = []
    for path in _active_python_files():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            lower_line = line.lower()
            matches = [pattern for pattern in FORBIDDEN_TFRECORD_REFERENCES if pattern.lower() in lower_line]
            if matches:
                relpath = path.relative_to(REPO_ROOT)
                violations.append(f"{relpath}:{line_number}: {', '.join(matches)}: {line.strip()}")

    assert not violations, "Active runtime TFRecord references found:\n" + "\n".join(violations)
