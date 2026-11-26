# Re-encode the Input Instead of Resizing Its Latent

Reference sampler, evaluation code and stock-clip identifiers for the paper

> **Re-encode the Input Instead of Resizing Its Latent: Training-Free Coarse-to-Fine Sampling for Few-Step Image-to-Video Animation**
> Luckison Du. AINIME Workshop at WACV 2027.

Coarse-to-fine image-to-video sampling runs early denoising steps at low resolution, so the image condition (the
first-frame *anchor*) must follow the changing resolution. The paper finds that the anchor should be built by
re-encoding the resized input image at every resolution, not by resizing an existing latent. **G3**
(448 → 576 → 704 pixel width, the first three of the model's four distilled noise levels, the anchor re-encoded
at each width) uses 47.7% of the native DiT compute of a public four-step Wan model, with no training.

## Contents

| Path | What it is |
|---|---|
| `g3_sampler.py` | Reference sampler: G3 and the other schedules of the paper (Table 4), with the anchor ablations (`--anchor upsample`, `--anchor downsample`) and the compute of Eq. 4 |
| `eval/metrics.py` | Input similarity (DINO ViT-B/16), sharpness (Laplacian variance) and motion (Farneback optical flow) |
| `eval/stats.py` | Clip-cluster paired bootstrap, sign test, net preference of the blind studies, recovery |
| `eval/VBENCH.md` | How the five VBench dimensions were run |
| `data/main42.csv` | The 42 inputs of the main evaluation set: 16 Pexels and 10 Pixabay videos, 8 Pexels photos, 8 generated stills (not released) |
| `data/val30.csv` | The 30 held-out clips of the 576-and-upscale study |
| `data/random100.csv` | The 100 randomly drawn held-out Pexels clips |

The CSV files list each input's source URL, author ID or photographer, horizontal crop offset (`crop_x`: the left edge of the
704-pixel-wide crop after the source video is scaled to a height of 1280 pixels) and the caption used for generation. A `_w1` / `_w2`
suffix marks two different windows cut from the same source video. The media themselves are not redistributed;
download them from Pexels or Pixabay under their licenses.

## Setup

```bash
pip install -r requirements.txt
```

Weights are downloaded from Hugging Face on first use:

- transformer: [`yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers`](https://huggingface.co/yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers) — the full Wan2.2-TI2V-5B-Turbo weights ([quanhaol/Wan2.2-TI2V-5B-Turbo](https://github.com/quanhaol/Wan2.2-TI2V-5B-Turbo)) in diffusers format;
- VAE, text encoder and tokenizer: [`Wan-AI/Wan2.2-TI2V-5B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers).

These are byte-identical (SHA-256) to the files used for the paper. Do not use the rank-64 Turbo LoRA instead of
the full transformer: it captures only part of the difference to the base model and four-step outputs degrade.

A 24 GB GPU is enough: the text encoder, the transformer and the VAE decoder are loaded one after another.

## Generate

```bash
python g3_sampler.py --image first_frame.png --prompt "The woman raises both arms and jumps." \
    --recipe G3 --seed 9001 --out g3.mp4
```

- `--recipe`: `G3` (448-576-704, 47.7%), `G3_full_middle` (448-704-704, 58.1%; recommended for stylized and animation inputs), `T4` (native four-step), `L3` (first three native steps), `two_step` (448-704), `low_res_576` (576x1024 output).
- `--anchor`: `reencode` (default, the paper's method), `upsample` or `downsample` (the ablations of Sec. 4.1).
- Output: 49 frames, 4 seconds at 12 fps, 704x1280 (width x height).

The image is resized to cover 704x1280 and center-cropped; the paper's inputs were already 704x1280. The anchors at
lower widths are made exactly as in the paper: the 704-wide anchor is decoded to pixels, resized with Lanczos and
encoded again. Noise for each width is subsampled from one full-resolution draw (s x s block averaging for integer
ratios, nearest sampling otherwise, then re-normalized), with a fresh draw for every transition. The paper decoded
clips with the TAEHV decoder used on the phone; this script decodes with the full Wan VAE.

## Reproducing the paper's numbers

Generate the four-second clips of an input set with two seeds (9001, 9002), score them with VBench (`eval/VBENCH.md`)
and `eval/metrics.py`, and compare samplers with `eval/stats.py`. The compute column of Table 4 is
`python -c "import g3_sampler as g; print(g.recipe_tflops('G3'), g.recipe_tflops('T4'))"` (286.6 and 601.2 TFLOPs).

The phone implementation, the generated videos and the raw blind ratings are not part of this release.

## Citation

```bibtex
@inproceedings{du2027reencode,
  title     = {Re-encode the Input Instead of Resizing Its Latent: Training-Free Coarse-to-Fine Sampling for Few-Step Image-to-Video Animation},
  author    = {Du, Luckison},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW)},
  year      = {2027}
}
```

## License

Code: Apache-2.0 (see `LICENSE`). Model weights, VBench, DINO and the listed stock media are under their own licenses.
