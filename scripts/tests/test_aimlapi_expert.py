from __future__ import annotations

import base64
import json

from aimlapi_expert import build_payload, parse_model_content
from PIL import Image


def test_build_payload_batches_frames_and_hides_image_bytes(tmp_path):
    image = Image.new("RGB", (20, 10), "blue")
    image_path = tmp_path / "frame.png"
    image.save(image_path)
    request = {
        "prompt": "Find the target",
        "frames": [{"frame_index": 0, "path": str(image_path)}],
    }

    payload = build_payload(
        model="xiaomi/mimo-v2.5",
        request=request,
        max_image_size=8,
        max_output_tokens=123,
    )

    assert payload["model"] == "xiaomi/mimo-v2.5"
    assert payload["max_tokens"] == 123
    assert len(payload["messages"][0]["content"]) == 2
    data_url = payload["messages"][0]["content"][1]["image_url"]["url"]
    assert data_url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1])


def test_parse_model_content_accepts_fenced_and_prose_wrapped_array():
    records = [{"frame_index": 0, "regions": []}]
    fenced = f"```json\n{json.dumps(records)}\n```"
    assert parse_model_content(fenced) == {"regions_by_frame": records}
    prose = f"Here it is:\n{json.dumps(records)}\nDone."
    assert parse_model_content(prose) == {"regions_by_frame": records}
