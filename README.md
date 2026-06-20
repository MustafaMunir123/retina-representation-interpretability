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