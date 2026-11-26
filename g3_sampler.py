#!/usr/bin/env python3
"""Reference sampler for "Re-encode the Input Instead of Resizing Its Latent: Training-Free
Coarse-to-Fine Sampling for Few-Step Image-to-Video Animation" (AINIME @ WACV 2027).

Training-free coarse-to-fine sampling for the public four-step Wan2.2-TI2V-5B-Turbo model.
At every resolution tier the first-frame condition (the "anchor") is built by re-encoding the
resized input image with the VAE, never by resizing an existing latent.

Recipes (widths of the 9:16 portrait output; noise levels = the model's distilled levels):
  G3              448 -> 576 -> 704   sigma 1.0, 0.93766, 0.83389   (47.7% of native DiT compute)
  G3_full_middle  448 -> 704 -> 704   same sigmas                   (58.1%; recommended for stylized input)
  T4              704 x 4             1.0, 0.93766, 0.83389, 0.62687 (native sampler, 100%)
  L3              704 x 3             first three native sigmas      (75.0%)
  two_step        448 -> 704          1.0, 0.93766                   (33.1%)
  low_res_576     576 x 3             first three native sigmas      (43.7%; output is 576x1024)
Anchor modes (for the paper's ablations): reencode (default, the paper's G3), upsample (the
first-tier anchor latent is bilinearly upsampled to later tiers), downsample (the full-resolution
anchor latent is resized down to earlier tiers).

Example:
  python g3_sampler.py --image input.png --prompt "A woman raises both arms and jumps." \
      --recipe G3 --seed 9001 --out g3.mp4
"""
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

TURBO_REPO = "yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers"   # full Turbo weights (diffusers format)
WAN_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"               # VAE, text encoder, tokenizer
NATIVE_SIGMAS = [1.0, 0.93766, 0.83389, 0.62687]           # Turbo's four distilled noise levels (timestep shift 5)
# width -> latent grid (height, width) of a 9:16 portrait frame; VAE stride 16
TIER = {352: (40, 22), 448: (50, 28), 512: (58, 32), 576: (64, 36), 640: (72, 40), 704: (80, 44)}
RECIPES = {
    "G3": ([448, 576, 704], NATIVE_SIGMAS[:3]),
    "G3_full_middle": ([448, 704, 704], NATIVE_SIGMAS[:3]),
    "T4": ([704] * 4, NATIVE_SIGMAS),
    "L3": ([704] * 3, NATIVE_SIGMAS[:3]),
    "two_step": ([448, 704], NATIVE_SIGMAS[:2]),
    "low_res_576": ([576] * 3, NATIVE_SIGMAS[:3]),
}
LATENT_FRAMES = 13          # 13 latent frames = 49 frames = 4 s at 12 fps (the paper's setting)
FPS = 12
DIM, FFN, TEXT_LEN, BLOCKS = 3072, 14336, 512, 30


# ----------------------------------------------------------------------------- models
def load_transformer(device, turbo=TURBO_REPO):
    from diffusers import WanTransformer3DModel
    return WanTransformer3DModel.from_pretrained(turbo, subfolder="transformer", torch_dtype=torch.bfloat16).to(device).eval()


def load_vae(device, wan=WAN_REPO):
    from diffusers import AutoencoderKLWan
    return AutoencoderKLWan.from_pretrained(wan, subfolder="vae", torch_dtype=torch.float32).to(device).eval()


@torch.no_grad()
def encode_text(prompt, device, wan=WAN_REPO):
    """UMT5 in bf16 on the GPU, padded to 512 tokens, positions after the prompt set to zero (as in the Wan
    pipeline). The encoder is freed afterwards so it never shares the GPU with the transformer."""
    from transformers import AutoTokenizer, UMT5EncoderModel
    tok = AutoTokenizer.from_pretrained(wan, subfolder="tokenizer")
    te = UMT5EncoderModel.from_pretrained(wan, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device).eval()
    ti = tok([prompt], padding="max_length", max_length=TEXT_LEN, truncation=True, add_special_tokens=True,
             return_attention_mask=True, return_tensors="pt")
    n = int(ti.attention_mask.sum())
    e = te(ti.input_ids.to(device), ti.attention_mask.to(device)).last_hidden_state
    e[:, n:] = 0
    del te
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return e.to(torch.bfloat16)


def _norm(vae, device):
    lm = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device)
    sd = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device)
    return lm, sd


def load_image(path, width=704, height=1280):
    """Resize to cover width x height (Lanczos) and center-crop. The paper's inputs were already 704x1280."""
    im = Image.open(path).convert("RGB")
    s = max(width / im.width, height / im.height)
    im = im.resize((max(width, round(im.width * s)), max(height, round(im.height * s))), Image.LANCZOS)
    x0, y0 = (im.width - width) // 2, (im.height - height) // 2
    return im.crop((x0, y0, x0 + width, y0 + height))


@torch.no_grad()
def encode_frame(vae, im, device):
    """PIL image -> normalized first-frame latent (1, 48, H/16, W/16), fp32 VAE, latent_dist.mode()."""
    lm, sd = _norm(vae, device)
    x = torch.from_numpy(np.asarray(im, np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None, :, None].to(device)
    return ((vae.encode(x).latent_dist.mode() - lm) / sd)[:, :, 0].float()


@torch.no_grad()
def tier_anchors(vae, z704, widths, device):
    """Per-tier re-encoding as in the paper: decode the 704-wide anchor to pixels, resize the image with
    Lanczos to each tier, and encode it again. Returns {latent grid (h, w): anchor latent (1, 48, h, w)}."""
    lm, sd = _norm(vae, device)
    out = {TIER[704]: z704}
    img = vae.decode(z704[:, :, None].to(device) * sd + lm).sample[0, :, 0].clamp(-1, 1)
    im = Image.fromarray(((img.permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).round().astype(np.uint8))
    for w in sorted(set(widths)):
        r = TIER[w]
        if r in out:
            continue
        x = torch.from_numpy(np.asarray(im.resize((r[1] * 16, r[0] * 16), Image.LANCZOS), np.float32) / 127.5 - 1.0)
        x = x.permute(2, 0, 1)[None, :, None].to(device)
        out[r] = ((vae.encode(x).latent_dist.mode() - lm) / sd)[:, :, 0].float()
    return out


# ----------------------------------------------------------------------------- resizing
def resize_noise(n, r):
    """Full-resolution white noise -> tier r. Integer ratio: s x s block average; otherwise nearest
    sampling. Then re-normalized, so the noise stays i.i.d. unit Gaussian."""
    if tuple(n.shape[-2:]) == tuple(r):
        return n
    B, C, T, H, W = n.shape
    mode = "area" if (H % r[0] == 0 and W % r[1] == 0) else "nearest-exact"
    m = F.interpolate(n.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W), size=r, mode=mode)
    m = m.reshape(B, T, C, *r).permute(0, 2, 1, 3, 4)
    return (m - m.mean()) / m.std()


def upsample_latent(x, r):
    """Bilinear resize of a video latent (B, C, T, H, W) in latent space (used for the video latent between steps)."""
    if tuple(x.shape[-2:]) == tuple(r):
        return x
    B, C, T, H, W = x.shape
    m = F.interpolate(x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W), size=r, mode="bilinear", align_corners=False)
    return m.reshape(B, T, C, *r).permute(0, 2, 1, 3, 4)


# ----------------------------------------------------------------------------- sampler
@torch.no_grad()
def sample(tr, anchors, text, recipe="G3", seed=9001, anchor_mode="reencode", latent_frames=LATENT_FRAMES, device="cuda"):
    """anchors: output of tier_anchors (must contain every tier of the recipe); text: output of encode_text.
    Returns the final clean latent (1, 48, latent_frames, h, w) in fp32 at the last tier."""
    widths, sigmas = RECIPES[recipe]
    tiers = [TIER[w] for w in widths]
    full = (1, 48, latent_frames, *TIER[704])
    first = anchors[tiers[0]]

    def anchor(r):
        if anchor_mode == "reencode" or tuple(r) == tuple(tiers[0]) and anchor_mode == "upsample":
            return anchors[tuple(r)]
        if anchor_mode == "upsample":       # ablation: resize the first-tier anchor latent instead of re-encoding
            return upsample_latent(first[:, :, None], r)[:, :, 0]
        if anchor_mode == "downsample":     # ablation: resize the full-resolution anchor latent
            z = anchors[TIER[704]]
            return z if tuple(r) == tuple(TIER[704]) else F.interpolate(z, size=r, mode="bilinear", antialias=True, align_corners=False)
        raise ValueError(anchor_mode)

    def pin(x, r):                          # first latent frame = clean anchor (timestep 0)
        x[:, :, 0] = anchor(r)
        return x

    gen = torch.Generator(device=device).manual_seed(seed)
    n0 = torch.randn(full, generator=gen, device=device)
    x = pin(resize_noise(n0, tiers[0]).clone(), tiers[0])
    for k, (r, s) in enumerate(zip(tiers, sigmas)):
        tpf = (r[0] // 2) * (r[1] // 2)                       # tokens per latent frame (1x2x2 patches)
        ts = torch.full((1, latent_frames * tpf), s * 1000.0, device=device)
        ts[:, :tpf] = 0.0                                     # anchor tokens get timestep 0
        v = tr(hidden_states=x.to(torch.bfloat16), timestep=ts, encoder_hidden_states=text, return_dict=False)[0].float()
        x0 = pin(x.float() - s * v, r)                        # rectified flow: x0 = x_sigma - sigma * v
        if k == len(tiers) - 1:
            return x0
        sn, rn = sigmas[k + 1], tiers[k + 1]
        nz = torch.randn(full, generator=gen, device=device)  # fresh noise for every transition
        x = pin((1 - sn) * upsample_latent(x0, rn) + sn * resize_noise(nz, rn), rn)


@torch.no_grad()
def decode(vae, z, device):
    lm, sd = _norm(vae, device)
    v = vae.decode(z.to(device) * sd + lm).sample[0].clamp(-1, 1)          # (3, T, H, W)
    return ((v.permute(1, 2, 3, 0).float().cpu().numpy() + 1) * 127.5).round().astype(np.uint8)


# ----------------------------------------------------------------------------- compute (Eq. 4 of the paper)
def block_flops(n_tokens):
    return 2 * (n_tokens * (6 * DIM ** 2 + 2 * DIM * FFN) + 2 * TEXT_LEN * DIM ** 2 + 2 * n_tokens ** 2 * DIM + 2 * n_tokens * TEXT_LEN * DIM)


def recipe_tflops(recipe, latent_frames=LATENT_FRAMES):
    widths, _ = RECIPES[recipe]
    return sum(BLOCKS * block_flops(latent_frames * (TIER[w][0] // 2) * (TIER[w][1] // 2)) for w in widths) / 1e12


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", required=True, help="first frame; resized to cover 704x1280 and center-cropped")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--recipe", default="G3", choices=list(RECIPES))
    ap.add_argument("--anchor", default="reencode", choices=["reencode", "upsample", "downsample"])
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--out", default="out.mp4")
    ap.add_argument("--save-latent", default="", help="optional .npy path for the final latent")
    ap.add_argument("--turbo", default=TURBO_REPO)
    ap.add_argument("--wan", default=WAN_REPO)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    text = encode_text(a.prompt, a.device, a.wan)
    vae = load_vae(a.device, a.wan)
    z704 = encode_frame(vae, load_image(a.image), a.device)
    anchors = tier_anchors(vae, z704, RECIPES[a.recipe][0] + [704], a.device)
    tr = load_transformer(a.device, a.turbo)
    z = sample(tr, anchors, text, a.recipe, a.seed, a.anchor, device=a.device)
    del tr
    if str(a.device).startswith("cuda"):
        torch.cuda.empty_cache()
    if a.save_latent:
        np.save(a.save_latent, z[0].cpu().numpy())
    frames = decode(vae, z, a.device)
    import imageio.v2 as imageio
    imageio.mimwrite(a.out, frames, fps=FPS, quality=9, macro_block_size=1)
    t, n = recipe_tflops(a.recipe), recipe_tflops("T4")
    print(f"{a.out}: {frames.shape[0]} frames {frames.shape[2]}x{frames.shape[1]}; DiT compute {t:.1f} TFLOPs = {100 * t / n:.1f}% of native T4")


if __name__ == "__main__":
    main()
