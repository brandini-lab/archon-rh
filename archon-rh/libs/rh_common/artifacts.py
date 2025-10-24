import json
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())
