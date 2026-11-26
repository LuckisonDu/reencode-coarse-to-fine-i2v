#!/usr/bin/env python3
"""Per-clip metrics used in the paper (besides VBench; see VBENCH.md).

  input_similarity  mean DINO ViT-B/16 cosine between frames 1-48 and a reference frame. The paper uses
                    frame 0 of the re-encoded (G3) clip, i.e. the decoded input image, as the reference for
                    both clips of a pair.
  sharpness         median Laplacian variance of frames 4-40 (grayscale). A proxy for high-frequency
                    content, not detail accuracy; the paper reports it as a ratio to T4 on the same clip and seed.
  motion            mean Farneback optical-flow magnitude between consecutive frames, frames resized to
                    176x320 (pixels/frame at 12 fps); also reported as a ratio to T4.

Example:
  python metrics.py --clip g3.mp4 --reference-clip g3.mp4 --t4 t4.mp4
"""
import argparse
import json
import subprocess

import cv2
import numpy as np
import torch
import torchvision.transforms as T

_DINO = {}


def read_frames(path, size=(704, 1280)):
    """RGB uint8 frames (N, H, W, 3), resized to size=(width, height) with bicubic if needed."""
    w, h = size
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-vf", f"scale={w}:{h}:flags=bicubic",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)[:49]


def dino_features(frames, device="cuda"):
    if "m" not in _DINO:
        _DINO["m"] = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").to(device).eval()
        _DINO["tf"] = T.Compose([T.Resize(224, antialias=False), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float().div(255).to(device)
    with torch.no_grad():
        return torch.nn.functional.normalize(_DINO["m"](_DINO["tf"](x)), dim=-1).cpu()


def input_similarity(frames, reference_frame, device="cuda"):
    ref = dino_features(reference_frame[None], device)
    return float((dino_features(frames[1:49], device) * ref).sum(-1).mean())


def sharpness(frames):
    return float(np.median([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F).var()
                            for f in frames[4:41]]))


def motion(frames):
    g = [cv2.cvtColor(cv2.resize(f, (176, 320)), cv2.COLOR_RGB2GRAY) for f in frames]
    return float(np.mean([np.linalg.norm(cv2.calcOpticalFlowFarneback(g[i - 1], g[i], None, 0.5, 3, 15, 3, 5, 1.2, 0), axis=2).mean()
                          for i in range(1, len(g))]))


def main():
    ap = argparse.ArgumentParser(description="Per-clip metrics of the paper")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--reference-clip", help="clip whose frame 0 is the reference for input similarity (paper: the G3 clip)")
    ap.add_argument("--t4", help="native T4 clip with the same input and seed, for sharpness and motion ratios")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    f = read_frames(a.clip)
    out = {"sharpness": sharpness(f), "motion": motion(f)}
    if a.reference_clip:
        out["input_similarity"] = input_similarity(f, read_frames(a.reference_clip)[0], a.device)
    if a.t4:
        t = read_frames(a.t4)
        out["sharpness_ratio_vs_T4"] = out["sharpness"] / sharpness(t)
        out["motion_ratio_vs_T4"] = out["motion"] / motion(t)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
