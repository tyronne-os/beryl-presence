"""SoulX-FlashHead (Lite) streaming renderer, driven through the repo's own streaming API
(flash_head.inference: get_pipeline / get_base_data / get_audio_embedding / run_pipeline), the same calls its
gradio_app_streaming.py makes. One chunk = (frame_num - motion_frames) frames = 0.96 s of audio at 25 fps."""
from __future__ import annotations

import os
import sys
import time

import numpy as np


class FlashHeadRenderer:
    def __init__(self, repo_dir: str, ckpt_dir: str, wav2vec_dir: str, cond_image: str, model_type: str = "lite", seed: int = 9999):
        self.repo_dir = os.path.abspath(repo_dir)
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        self.wav2vec_dir = os.path.abspath(wav2vec_dir)
        self.cond_image = os.path.abspath(cond_image)
        self.model_type, self.seed = model_type, seed
        self.pipe = None

    def load(self) -> float:
        t0 = time.perf_counter()
        # flash_head.inference opens its YAML with a relative path at import time
        os.chdir(self.repo_dir)
        sys.path.insert(0, self.repo_dir)
        import torch
        from flash_head import inference as fh

        self._torch, self._fh = torch, fh
        self.pipe = fh.get_pipeline(world_size=1, ckpt_dir=self.ckpt_dir, model_type=self.model_type, wav2vec_dir=self.wav2vec_dir)
        fh.get_base_data(self.pipe, cond_image_path_or_dir=self.cond_image, base_seed=self.seed, use_face_crop=False)
        p = fh.get_infer_params()  # read after get_pipeline: it fills in motion_frames_num
        self.sr, self.fps = p["sample_rate"], p["tgt_fps"]
        self.frame_num, self.motion = p["frame_num"], p["motion_frames_num"]
        self.slice_frames = self.frame_num - self.motion
        self.slice_samples = self.slice_frames * self.sr // self.fps
        self._ctx_len = self.sr * p["cached_audio_duration"]
        self._end = p["cached_audio_duration"] * self.fps
        self._start = self._end - self.frame_num
        self.reset()
        return time.perf_counter() - t0

    def reset(self):
        self._ctx = np.zeros(self._ctx_len, dtype=np.float32)

    @property
    def chunk_seconds(self) -> float:
        return self.slice_samples / self.sr

    def render(self, audio_slice: np.ndarray) -> tuple[np.ndarray, float]:
        """audio_slice: float32 16 kHz, exactly slice_samples long -> (frames uint8 [T,H,W,3] RGB, GPU seconds)."""
        assert len(audio_slice) == self.slice_samples, (len(audio_slice), self.slice_samples)
        torch, fh = self._torch, self._fh
        self._ctx = np.concatenate((self._ctx[len(audio_slice):], audio_slice.astype(np.float32)))
        emb = fh.get_audio_embedding(self.pipe, self._ctx, self._start, self._end)
        torch.cuda.synchronize()
        t = time.perf_counter()
        video = fh.run_pipeline(self.pipe, emb)[self.motion:]
        frames = video.clamp(0, 255).to(torch.uint8).cpu().numpy()
        torch.cuda.synchronize()
        return frames, time.perf_counter() - t
