# Retina Subspace Audit

Lightweight audit for frozen retinal fundus embeddings. The project tests
whether diabetic-retinopathy representation directions are entangled with
image-quality and preprocessing bottlenecks, then evaluates one small
improvement such as cropping, matched direction extraction, or bottleneck
subspace residualization.

This is a representation audit and data-engineering project, not a diabetic
retinopathy classifier benchmark.

## Phase 0 Status

Phase 0 establishes the reproducible project scaffold:

- source package under `src/retina_audit/`
- runnable script entry points under `scripts/`
- dataset/model configs under `configs/`
- stable output locations under `outputs/`
- paper skeleton under `paper/`
- structured planning docs in the separate docs repo under `docs/`

## Phase 1 Status

Phase 1 implements dataset access and manifest auditing.

Run the primary EyePACS-style audit after downloading data to the paths in the
config:

```bash
python scripts/00_audit_dataset.py --config configs/eyepacs_resized_dinov2.yaml
```

Expected outputs:

- `outputs/manifests/dataset_summary.json`
- `outputs/manifests/image_manifest.parquet`
- `outputs/manifests/label_counts.csv`
- `outputs/manifests/missing_files.csv`
- `outputs/manifests/corrupt_images.csv`
- `outputs/manifests/filename_parse_report.json`

The audit supports:

- CSV-label DR severity datasets such as EyePACS-style resized data and APTOS
- folder-label disease-category datasets such as RF50K

## Phase 2 Status

Phase 2 implements CPU-side pixel-derived bottleneck features from the Phase 1
manifest.

Smoke test:

```bash
python scripts/01_compute_quality_features.py \
  --manifest outputs/manifests/image_manifest.parquet \
  --limit 100
```

Full run:

```bash
python scripts/01_compute_quality_features.py \
  --manifest outputs/manifests/image_manifest.parquet
```

Expected outputs:

- `outputs/quality/quality_features.parquet`
- `outputs/quality/quality_summary.json`
- `outputs/figures/quality/histograms/*.png`
- `outputs/figures/quality/sample_grids/*.png`

The Phase 2 gate passes when at least three bottleneck features have meaningful
spread and the minimum set of sharpness, brightness, and contrast is usable.

## Phase 3 Status

Phase 3 implements frozen embedding extraction.

Local CPU smoke test only:

```bash
python scripts/02_extract_embeddings.py \
  --config configs/eyepacs_resized_timm.yaml \
  --subset 8 \
  --allow-cpu
```

GPU run:

```bash
python scripts/02_extract_embeddings.py \
  --config configs/eyepacs_resized_dinov2.yaml \
  --subset 2000 \
  --device cuda
```

Kaggle run with portable manifest paths:

```bash
python scripts/02_extract_embeddings.py \
  --config configs/eyepacs_resized_dinov2.yaml \
  --manifest outputs/manifests/image_manifest.parquet \
  --image-root /kaggle/input/diabetic-retinopathy-resized \
  --subset 2000 \
  --device cuda
```

Required progression:

1. `--subset 2000`
2. `--subset 10000`
3. `--subset all`

Expected outputs:

- `outputs/embeddings/{dataset}_{model}_{preprocess}_embeddings.npy`
- `outputs/embeddings/{dataset}_{model}_{preprocess}_index.parquet`
- `outputs/embeddings/{dataset}_{model}_{preprocess}_meta.json`

The canonical first run is `configs/eyepacs_resized_dinov2.yaml`. Use
`configs/eyepacs_cropped_dinov2.yaml` for the preprocessing comparison after the
baseline is validated.

## Phase 4 Status

Phase 4 implements linear probe sanity checks for disease and bottleneck signal
in cached embeddings.

Run probes on a 10k embedding artifact:

```bash
python scripts/04_run_probes.py \
  --embeddings outputs/embeddings/eyepacs_resized_dinov2_resized_10k_embeddings.npy \
  --index outputs/embeddings/eyepacs_resized_dinov2_resized_10k_index.parquet \
  --quality outputs/quality/quality_features.parquet \
  --output-prefix eyepacs_resized_dinov2_resized_10k
```

Expected outputs:

- `outputs/probes/{run}_probe_metrics.csv`
- `outputs/probes/{run}_probe_coefficients.csv`
- `outputs/probes/{run}_probe_predictions.parquet`
- `outputs/probes/{run}_probe_meta.json`
- `outputs/tables/probe_metrics.csv`
- `outputs/tables/probe_comparison.csv`
- `outputs/figures/probes/{run}_probe_auc.png`

## Hugging Face Data Sync

Raw data and generated data artifacts are not committed to git. To mirror them
through the Hugging Face dataset repo while preserving project-relative paths:

```bash
python scripts/sync_hf_dataset.py upload \
  --repo-id mm2036/retina-representation-interpretability
```

On another machine:

```bash
python scripts/sync_hf_dataset.py download \
  --repo-id mm2036/retina-representation-interpretability
```

The default sync includes processed artifacts under `outputs/manifests`,
`outputs/quality`, `outputs/embeddings`, `outputs/probes`, `outputs/tables`, and
the quality/probe figure folders. Raw images should come from the original
Kaggle dataset, not this Hugging Face mirror.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Smoke check:

```bash
python -c "import retina_audit; print(retina_audit.__version__)"
python scripts/00_audit_dataset.py --help
```

## Canonical Artifacts

All downstream analysis must key by `image_id`, not row position alone.

Required shared outputs:

- `outputs/manifests/image_manifest.parquet`
- `outputs/quality/quality_features.parquet`
- `outputs/embeddings/{dataset}_{model}_{preprocess}_embeddings.npy`
- `outputs/embeddings/{dataset}_{model}_{preprocess}_index.parquet`
- `outputs/embeddings/{dataset}_{model}_{preprocess}_meta.json`
- `outputs/tables/*.csv`

Every embedding `.npy` must have a matching index file in identical row order
and a metadata JSON recording dataset, model, checkpoint, preprocessing,
embedding dimension, image count, failures, runtime, and compute environment.

## Claim Discipline

Use this framing:

- representation audit
- bottleneck diagnosis
- frozen-embedding workflow
- deconfounded contrastive direction analysis

Avoid:

- clinical deployment claims
- demographic fairness claims without demographic metadata
- state-of-the-art classifier framing
- causal claims from direction removal