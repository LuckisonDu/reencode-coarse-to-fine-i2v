# VBench scores

The paper reports five [VBench](https://github.com/Vchitect/VBench) dimensions in custom-input mode:
subject consistency, background consistency, aesthetic quality, imaging quality and dynamic degree.
Each dimension was run separately on a folder of clips (one folder per sampler and seed):

```bash
for dim in subject_consistency background_consistency aesthetic_quality imaging_quality dynamic_degree; do
  vbench evaluate --videos_path clips/G3_sb9001 --dimension $dim --mode custom_input --output_path vb/G3_sb9001
done
```

Notes on how the numbers in the paper were produced:

- Clips are scored at the generated 12 fps (49 frames, 704x1280); the 576-wide sampler is upscaled to 704x1280 with bicubic interpolation before scoring.
- Imaging quality is reported on a 0-1 scale (VBench's value divided by 100).
- VBench's subject consistency uses frame 0 as a reference; with a pinned first frame the paper also reports the gap measured from frame 1 (Sec. 4.1).
- Paired differences use `stats.paired_cluster_bootstrap` with keys `(seed, clip)`.
