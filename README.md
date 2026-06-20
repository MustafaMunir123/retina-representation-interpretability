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