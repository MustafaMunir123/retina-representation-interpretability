# Data Directory

Do not commit raw medical images or downloaded Kaggle datasets.

Expected local layout:

```text
data/
  raw/
  metadata/
  manifests/
  splits/
```

Tracked project artifacts should start from audited metadata, manifests, and
split files. Raw data paths are configured in `configs/*.yaml`.
