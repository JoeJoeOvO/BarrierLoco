import ast
import pathlib
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("iteration", [543, 3543])
def test_completed_video_uses_current_iteration_including_resume(tmp_path, iteration):
    path = pathlib.Path(__file__).resolve().parents[1] / "locolab/rsl_rl/utils/logger.py"
    cls = next(
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "Logger"
    )
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_upload_pending_videos"
    )
    namespace = {"pathlib": pathlib}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    video = tmp_path / "rl-video-step-12000.mp4"
    video.write_bytes(b"completed clip")
    uploads = []
    logger = SimpleNamespace(
        writer=SimpleNamespace(log_video=lambda *args, **kwargs: uploads.append(kwargs)),
        disable_logs=False,
        logger_type="wandb",
        video_sources=[{"path": str(tmp_path), "tag": "Videos/train", "fps": 50, "steps_per_iter": 24}],
        uploaded_videos=set(),
        observed_video_states={},
    )
    upload = namespace["_upload_pending_videos"]
    upload(logger, iteration)
    assert not uploads  # Wait for an unchanged file size/mtime before uploading.
    upload(logger, iteration)
    assert uploads == [{"tag": "Videos/train", "global_step": iteration, "fps": 50}]
    upload(logger, iteration + 1)
    assert len(uploads) == 1
