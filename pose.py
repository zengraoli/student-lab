"""姿势提取：用 rtmlib（RTMW / DWPose 全身 133 点）把照片转成 OpenPose 风格骨架图。

骨架图（黑底彩色火柴人，含手和脸的关键点）可以直接作为 Z-Image-Turbo ControlNet Union 的
control image，也可以作为 Qwen-Image-2.1 编辑时的姿势参考图。模型文件首次使用时由 rtmlib
自动下载到 ~/.cache，CPU 推理，单张约 1–3 秒，不占显存。
"""
from __future__ import annotations

import threading

import numpy as np
from PIL import Image

_model = None
_lock = threading.Lock()


def _detector():
    global _model
    with _lock:
        if _model is None:
            from rtmlib import Wholebody
            _model = Wholebody(to_openpose=True, mode='balanced', backend='onnxruntime', device='cpu')
    return _model


def looks_like_skeleton(image: Image.Image) -> bool:
    """已经是骨架图（黑底、少量高饱和彩色线条）就直接用，不再重复提取。"""
    small = np.asarray(image.convert('RGB').resize((256, 256)), dtype=np.int16)
    dark = (small.max(axis=2) < 24).mean()
    colored = ((small.max(axis=2) - small.min(axis=2)) > 80).mean()
    return dark > 0.80 and 0.002 < colored < 0.2


def extract(image: Image.Image, max_side: int = 1024) -> tuple[Image.Image, dict]:
    """返回 (骨架图, 信息)。骨架图尺寸与缩放后的输入一致（最长边 max_side）。"""
    from rtmlib import draw_skeleton

    rgb = image.convert('RGB')
    scale = min(1.0, max_side / max(rgb.size))
    if scale < 1.0:
        rgb = rgb.resize((max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))), Image.Resampling.LANCZOS)
    bgr = np.ascontiguousarray(np.array(rgb)[:, :, ::-1])
    keypoints, scores = _detector()(bgr)
    canvas = np.zeros_like(bgr)
    people = 0 if keypoints is None else int(len(keypoints))
    if people:
        canvas = draw_skeleton(canvas, keypoints, scores, openpose_skeleton=True, kpt_thr=0.4)
    skeleton = Image.fromarray(np.ascontiguousarray(canvas[:, :, ::-1]))
    visible = 0
    if people:
        visible = int((np.asarray(scores) > 0.4).sum())
    return skeleton, {'people': people, 'visible_keypoints': visible, 'size': list(skeleton.size)}
