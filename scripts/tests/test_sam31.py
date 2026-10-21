from __future__ import annotations

import numpy as np

from samurai.media import VideoInfo, write_gray_video
from samurai.sam31 import SegmentationPrompt, segment_video


class FakePredictor:
    def __init__(self):
        self.requests = []

    def handle_request(self, request):
        self.requests.append(request)
        return {"session_id": "session"}

    def handle_stream_request(self, request):
        self.requests.append(request)
        for index in range(2):
            yield {
                "frame_index": index,
                "outputs": {
                    "out_binary_masks": np.zeros((1, 4, 6), dtype=bool),
                    "out_obj_ids": np.array([0]),
                },
            }


def test_prompt_normalizes_absolute_coordinates():
    prompt = SegmentationPrompt(
        frame_index=3,
        points=((10, 20),),
        point_labels=(1,),
        boxes_xyxy=((10, 20, 30, 60),),
    )
    request = prompt.request(VideoInfo(width=100, height=200, fps=30, frames=3))
    assert request["points"] == [[0.1, 0.1]]
    assert request["point_labels"] == [1]
    assert request["bounding_boxes"] == [[0.1, 0.1, 0.2, 0.2]]
    assert request["bounding_box_labels"] == [True]


def test_segment_video_writes_all_frames_and_closes_session(tmp_path):
    video = tmp_path / "video.mp4"
    write_gray_video([np.zeros((4, 6), dtype=np.uint8) for _ in range(2)], video, fps=10)
    predictor = FakePredictor()
    result = segment_video(
        predictor,
        video,
        SegmentationPrompt(frame_index=0, text="cup"),
        tmp_path / "mask.mp4",
    )
    assert result["frames"] == 2
    assert result["model_version"] == "sam3.1"
    assert (tmp_path / "mask.mp4").is_file()
    assert predictor.requests[-1]["type"] == "close_session"
