#!/usr/bin/env python3
"""Adapter from an AIMLAPI multimodal model to samurai expert regions."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_ENDPOINT = "https://api.aimlapi.com/v1/chat/completions"


def _image_data_url(path: Path, max_image_size: int) -> str:
    image = Image.open(path).convert("RGB")
    if max(image.size) > max_image_size:
        scale = max_image_size / max(image.size)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_payload(
    *,
    model: str,
    request: dict[str, Any],
    max_image_size: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    prompt = request.get("prompt") or "Identify the target object to remove."
    contract = (
        "Return only a JSON array named regions_by_frame. For each input image, "
        'in the same order, emit {"frame_index": <integer>, "regions": [...]}. '
        'Each region is {"type": "box" or "polygon", "coordinates": [...], '
        '"confidence": <0..1>}. Box coordinates are normalized x1,y1,x2,y2. '
        "Polygon coordinates are normalized x,y pairs. No prose."
    )
    content: list[dict[str, Any]] = [
        {"type": "text", "text": f"{prompt}\n\n{contract}"},
    ]
    for frame in request["frames"]:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(Path(frame["path"]), max_image_size)},
            }
        )
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_output_tokens,
        "temperature": 0,
    }


def parse_model_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("model response does not contain a JSON array")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("regions_by_frame must be an array")
    return {"regions_by_frame": parsed}


def call_api(
    payload: dict[str, Any], *, endpoint: str, api_key: str, timeout: float
) -> dict[str, Any]:
    http_request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"AIMLAPI request failed: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--max-image-size", type=int, default=1280)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    try:
        request = json.loads(sys.stdin.read())
        payload = build_payload(
            model=args.model,
            request=request,
            max_image_size=args.max_image_size,
            max_output_tokens=args.max_output_tokens,
        )
        api_key = os.environ.get("AIMLAPI_KEY")
        if not api_key:
            raise RuntimeError("AIMLAPI_KEY is not set")
        response = call_api(
            payload,
            endpoint=args.endpoint,
            api_key=api_key,
            timeout=args.timeout,
        )
        content = response["choices"][0]["message"]["content"]
        result = parse_model_content(content)
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as error:
        print(f"aimlapi_expert: {error}", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
