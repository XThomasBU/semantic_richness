# Linear Probe — Character Holdout

Same angle linear-probe task as the standard probe, but **test characters are completely held out** from training (20% by default). Training uses **all strokes** of train characters; testing uses **all strokes** of held-out characters.

## Reuses existing features

No re-precompute needed. Point at caches under:

`results/linear_probe/features/*_images_all_all_strokes_features.pt`

## Run

```bash
python -m experiments.linear_probe.run_probe_char_holdout \
  --features_path results/linear_probe/features/siglip_images_all_all_strokes_features.pt \
  --output_dir results/linear_probe_char_holdout
```

All encoders (SGE):

```bash
qsub experiments/linear_probe/run_probe_char_holdout.sh
```

## Outputs

- `{encoder}_char_holdout_split.json` — train/test character lists
- `{encoder}_angle_probe_char_holdout_{diff,concat}.json` — metrics + plots

## vs standard linear probe (first_stroke_test)

| | Standard probe | Character holdout |
|---|---|---|
| Test units | First stroke, **all** characters | **All strokes**, **held-out** characters only |
| Train | Non-first strokes | All strokes of train chars |
| Generalization | Stroke variant | New character identity |
