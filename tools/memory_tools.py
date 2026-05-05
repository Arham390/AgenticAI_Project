import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import chromadb
except Exception:
    chromadb = None


_CHROMA_PATH = Path(__file__).resolve().parents[1] / "outputs" / "chroma_db"
_COLLECTION = None
if chromadb is not None:
    try:
        _CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _persistent = chromadb.PersistentClient(path=str(_CHROMA_PATH))
        _COLLECTION = _persistent.get_or_create_collection("memory")
    except Exception:
        try:
            _ephemeral = chromadb.Client()
            _COLLECTION = _ephemeral.get_or_create_collection("memory")
        except Exception:
            _COLLECTION = None


_FALLBACK_STORE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "memory_store.json"


def _stable_id(data: Any) -> str:
    payload = json.dumps(data, default=str, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _commit_to_file(data: Any) -> str:
    _FALLBACK_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = []
    if _FALLBACK_STORE_PATH.exists():
        try:
            loaded = json.loads(_FALLBACK_STORE_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                records = loaded
        except Exception:
            records = []

    records.append({"id": _stable_id(data), "data": data})
    _FALLBACK_STORE_PATH.write_text(
        json.dumps(records, indent=2, default=str),
        encoding="utf-8",
    )
    return "stored:file"

def commit_memory(data: Any) -> str:
    if _COLLECTION is None:
        return _commit_to_file(data)

    doc_id = _stable_id(data)
    payload = json.dumps(data, default=str)

    try:
        _COLLECTION.upsert(documents=[payload], ids=[doc_id])
        return "stored:chromadb"
    except Exception:
        return _commit_to_file(data)