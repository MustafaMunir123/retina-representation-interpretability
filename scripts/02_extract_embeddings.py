#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retina_audit.config import load_config, require_sections
from retina_audit.embeddings import extract_embeddings_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract frozen image embeddings.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--subset", default="2000", help="Image subset size or 'all'.")
    parser.add_argument(
        "--manifest",
        default="outputs/manifests/image_manifest.parquet",
        help="Path to Phase 1 image manifest.",
    )
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda, mps, cpu.")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional artifact prefix overriding {dataset}_{model}_{preprocess}.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Override dataset image root for resolving relative_image_path.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU extraction for tiny smoke tests only.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    require_sections(config, ("dataset", "preprocess", "model", "outputs"))
    metadata = extract_embeddings_from_config(
        config,
        manifest_path=args.manifest,
        subset=args.subset,
        device=args.device,
        output_prefix=args.output_prefix,
        allow_cpu=args.allow_cpu,
        image_root=args.image_root,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"\nEmbedding extraction complete: {metadata['num_images']} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
