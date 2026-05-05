"""
Local ComfyUI HTTP API client (txt2img).

Prereqs:
  - ComfyUI running (default http://127.0.0.1:8188)
  - A checkpoint file name exactly as listed in ComfyUI's checkpoint loader

Env:
  COMFYUI_URL            Base URL (default http://127.0.0.1:8188)
  COMFYUI_CHECKPOINT     Required to enable real generation, e.g. v1-5-pruned-emaonly.safetensors
  COMFYUI_NEGATIVE       Optional default negative prompt
  COMFYUI_WIDTH          Default 512
  COMFYUI_HEIGHT         Default 512
  COMFYUI_STEPS          Default 28
  COMFYUI_CFG            Default 7.0 (passed as float in JSON)
  COMFYUI_TIMEOUT_SEC    Max wait for job (default 300)
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _env_str(name: str, default: str) -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def comfyui_configured() -> bool:
    return bool(_env_str("COMFYUI_CHECKPOINT", ""))


def _build_txt2img_workflow(
    *,
    positive: str,
    negative: str,
    checkpoint: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    filename_prefix: str,
) -> Dict[str, Any]:
    """Minimal KSampler graph compatible with stock ComfyUI (no custom nodes)."""
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive, "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["4", 1]},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": filename_prefix, "images": ["8", 0]},
        },
    }


def _post_json(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI POST {url} failed ({exc.code}): {detail}") from exc


def _get_json(url: str, timeout: int = 60) -> Any:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI GET {url} failed ({exc.code}): {detail}") from exc


def _queue_prompt(base: str, workflow: Dict[str, Any]) -> str:
    client_id = str(uuid.uuid4())
    body = {"prompt": workflow, "client_id": client_id}
    out = _post_json(f"{base}/prompt", body, timeout=120)
    if out.get("node_errors"):
        raise RuntimeError(f"ComfyUI workflow node_errors: {out['node_errors']}")
    pid = out.get("prompt_id")
    if not pid:
        raise RuntimeError(f"ComfyUI /prompt did not return prompt_id: {out}")
    return str(pid)


def _fetch_history_for_prompt(base: str, prompt_id: str) -> Optional[Dict[str, Any]]:
    """ComfyUI returns a dict keyed by prompt_id (GET /history)."""
    try:
        hist = _get_json(f"{base}/history", timeout=60)
    except urllib.error.HTTPError:
        return None
    if not isinstance(hist, dict):
        return None
    entry = hist.get(prompt_id)
    return entry if isinstance(entry, dict) else None


def _first_output_image(entry: Dict[str, Any]) -> Optional[Dict[str, str]]:
    outputs = entry.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for node_out in outputs.values():
        if not isinstance(node_out, dict):
            continue
        images = node_out.get("images")
        if not isinstance(images, list) or not images:
            continue
        first = images[0]
        if isinstance(first, dict) and first.get("filename"):
            return {
                "filename": str(first["filename"]),
                "subfolder": str(first.get("subfolder") or ""),
                "type": str(first.get("type") or "output"),
            }
    return None


def _wait_for_output(
    base: str, prompt_id: str, timeout_sec: int
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    deadline = time.monotonic() + timeout_sec
    last_entry: Optional[Dict[str, Any]] = None
    while time.monotonic() < deadline:
        entry = _fetch_history_for_prompt(base, prompt_id)
        last_entry = entry or last_entry
        if entry:
            status = entry.get("status")
            if isinstance(status, dict):
                msgs = status.get("messages")
                if isinstance(msgs, list):
                    for m in msgs:
                        if isinstance(m, list) and len(m) >= 2:
                            if m[0] == "execution_error":
                                raise RuntimeError(f"ComfyUI execution_error: {m[1]}")
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI job failed: {status}")
            img = _first_output_image(entry)
            if img:
                return entry, img
        time.sleep(0.35)
    raise TimeoutError(
        f"ComfyUI timed out after {timeout_sec}s waiting for prompt_id={prompt_id}. "
        f"Last entry: {last_entry!r}"
    )


def _download_view(base: str, image_info: Dict[str, str], dest: Path) -> None:
    q = urllib.parse.urlencode(
        {
            "filename": image_info["filename"],
            "subfolder": image_info["subfolder"],
            "type": image_info["type"],
        }
    )
    url = f"{base}/view?{q}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    dest.write_bytes(data)


def generate_character_image_via_comfyui(positive_prompt: str, dest_png: Path, seed: int | None = None) -> str:
    """
    Run txt2img on local ComfyUI and save the first output image to dest_png.
    Returns path as string (may differ extension if Comfy returns non-png — we still write bytes).
    """
    base = _env_str("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
    ckpt = _env_str("COMFYUI_CHECKPOINT", "")
    if not ckpt:
        raise RuntimeError("COMFYUI_CHECKPOINT is not set")

    neg = _env_str(
        "COMFYUI_NEGATIVE",
        "low quality, blurry, deformed hands, extra fingers, watermark, text, logo",
    )
    w = max(64, min(2048, _env_int("COMFYUI_WIDTH", 512)))
    h = max(64, min(2048, _env_int("COMFYUI_HEIGHT", 512)))
    steps = max(1, min(150, _env_int("COMFYUI_STEPS", 28)))
    cfg = max(1.0, min(30.0, _env_float("COMFYUI_CFG", 7.0)))
    timeout = max(30, _env_int("COMFYUI_TIMEOUT_SEC", 300))

    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    prefix = f"A3_{dest_png.stem}"[:120] or "A3_character"

    wf = _build_txt2img_workflow(
        positive=positive_prompt[:2000],
        negative=neg[:2000],
        checkpoint=ckpt,
        width=w,
        height=h,
        steps=steps,
        cfg=cfg,
        seed=seed,
        filename_prefix=prefix,
    )

    prompt_id = _queue_prompt(base, wf)
    _, img_info = _wait_for_output(base, prompt_id, timeout)

    ext = Path(img_info["filename"]).suffix.lower() or ".png"
    out_path = dest_png.with_suffix(ext)
    _download_view(base, img_info, out_path)

    sidecar = out_path.parent / f"{out_path.stem}_prompt.txt"
    sidecar.write_text(
        f"comfy_prompt_id={prompt_id}\nseed={seed}\npositive=\n{positive_prompt}\n",
        encoding="utf-8",
    )
    return str(out_path.resolve())
