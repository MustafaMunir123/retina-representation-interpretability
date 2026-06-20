#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


DEFAULT_REPO_ID = "mm2036/retina-representation-interpretability"
DEFAULT_MANIFEST_PATH = "_retina_project_file_manifest.json"
DEFAULT_ARCHIVE_DIR = "_archives"
DEFAULT_INCLUDE_PATHS = (
    "outputs/manifests",
    "outputs/quality",
    "outputs/figures/quality",
)
EXCLUDED_NAMES = {".DS_Store", ".gitkeep"}


@dataclass(frozen=True)
class FileRecord:
    local_path: Path
    repo_path: str
    size_bytes: int
    sha256: str


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload/download project data to a Hugging Face dataset repo."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload = subparsers.add_parser("upload", help="Upload required data files to HF dataset.")
    add_common_args(upload)
    upload.add_argument(
        "--include",
        nargs="*",
        default=list(DEFAULT_INCLUDE_PATHS),
        help="Project-root-relative files/directories to upload.",
    )
    upload.add_argument("--dry-run", action="store_true", help="Print planned upload only.")
    upload.add_argument(
        "--commit-message",
        default="Sync retina project data artifacts",
        help="HF dataset commit message.",
    )
    upload.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Parallel workers for Hugging Face large-folder upload in files mode.",
    )
    upload.add_argument(
        "--mode",
        choices=("archive", "files"),
        default="files",
        help="Upload exact individual files or archive shards. Use archive for large raw-data mirrors.",
    )
    upload.add_argument(
        "--archive-max-bytes",
        type=int,
        default=2_000_000_000,
        help="Maximum uncompressed payload per tar archive shard.",
    )
    upload.add_argument(
        "--force",
        action="store_true",
        help="Re-upload files even if the same repo path already exists remotely.",
    )

    download = subparsers.add_parser(
        "download", help="Download HF dataset files into project-relative paths."
    )
    add_common_args(download)
    download.add_argument("--dry-run", action="store_true", help="Print planned download only.")
    download.add_argument(
        "--force",
        action="store_true",
        help="Download and overwrite local files even when SHA-256 already matches.",
    )

    args = parser.parse_args()
    if args.command == "upload":
        upload_dataset(args)
    elif args.command == "download":
        download_dataset(args)
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo id.")
    parser.add_argument(
        "--project-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Project root used for relative path mapping.",
    )
    parser.add_argument(
        "--manifest-path",
        default=DEFAULT_MANIFEST_PATH,
        help="Repo path for the generated/consumed mapping manifest.",
    )
    parser.add_argument(
        "--token-env",
        default=None,
        help="Optional env var name containing a Hugging Face token.",
    )


def upload_dataset(args: argparse.Namespace) -> None:
    api, token = hf_api(args.token_env)
    project_root = args.project_root.resolve()
    records = collect_records(project_root, args.include)
    manifest = build_manifest(records, storage_mode=args.mode)
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")

    print(f"Repo: {args.repo_id}")
    print(f"Project root: {project_root}")
    print(f"Storage mode: {args.mode}")
    print(f"Files represented: {len(records)}")
    print(f"Total bytes: {sum(record.size_bytes for record in records):,}")
    if args.dry_run:
        remote_paths = remote_repo_paths(api, args.repo_id, token)
        pending_records = records if args.force else skip_existing_remote(records, remote_paths)
        skipped = len(records) - len(pending_records)
        if args.mode == "files":
            print(f"Already present remotely: {skipped}")
            print(f"Files that would upload: {len(pending_records)}")
        else:
            shards = plan_archive_shards(records, args.archive_max_bytes)
            print(f"Archive shards that would be created: {len(shards)}")
        for record in records[:20]:
            action = "UPLOAD" if args.force or record.repo_path not in remote_paths else "SKIP"
            if args.mode == "files":
                print(f"{action} {record.local_path} -> {record.repo_path}")
            else:
                print(f"PACK {record.local_path} -> {record.repo_path}")
        if len(records) > 20:
            print(f"... {len(records) - 20} more files")
        print(f"UPLOAD manifest -> {args.manifest_path}")
        return

    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=True, exist_ok=True, token=token)
    if args.mode == "archive":
        upload_archives(api, token, args, records)
        return

    remote_paths = remote_repo_paths(api, args.repo_id, token)
    pending_records = records if args.force else skip_existing_remote(records, remote_paths)
    skipped = len(records) - len(pending_records)
    print(f"Already present remotely: {skipped}")
    print(f"Files to upload now: {len(pending_records)}")
    with tempfile.TemporaryDirectory(prefix="retina_hf_upload_") as staging_dir:
        staging_root = Path(staging_dir)
        for record in pending_records:
            destination = staging_root / record.repo_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            link_or_copy(record.local_path, destination)
        manifest_destination = staging_root / args.manifest_path
        manifest_destination.parent.mkdir(parents=True, exist_ok=True)
        manifest_destination.write_bytes(manifest_bytes)
        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=staging_root,
            private=True,
            num_workers=args.num_workers,
        )
    print(f"uploaded {len(pending_records)} files plus manifest {args.manifest_path}")


def download_dataset(args: argparse.Namespace) -> None:
    _, token = hf_api(args.token_env)
    from huggingface_hub import hf_hub_download

    project_root = args.project_root.resolve()
    manifest_file = hf_hub_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        filename=args.manifest_path,
        token=token,
    )
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    storage_mode = manifest.get("storage_mode", "files")
    print(f"Repo: {args.repo_id}")
    print(f"Project root: {project_root}")
    print(f"Storage mode: {storage_mode}")
    print(f"Files to download: {len(files)}")
    if args.dry_run:
        local_present = 0
        local_current = 0
        for entry in files:
            destination = safe_project_path(project_root, entry["relative_path"])
            if destination.exists():
                local_present += 1
                if sha256_file(destination) == entry["sha256"]:
                    local_current += 1
        print(f"Local files present: {local_present}")
        print(f"Local files already current: {local_current}")
        print(f"Files that would download: {len(files) - local_current if not args.force else len(files)}")
        for entry in files[:20]:
            destination = project_root / entry["relative_path"]
            action = (
                "DOWNLOAD"
                if args.force
                or not destination.exists()
                or sha256_file(destination) != entry["sha256"]
                else "SKIP"
            )
            print(f"{action} {entry['repo_path']} -> {destination}")
        if len(files) > 20:
            print(f"... {len(files) - 20} more files")
        return

    if storage_mode == "archive":
        download_archives(args, token, manifest, files, project_root)
        return

    for entry in files:
        repo_path = entry["repo_path"]
        destination = safe_project_path(project_root, entry["relative_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not args.force and destination.exists() and sha256_file(destination) == entry["sha256"]:
            print(f"exists {entry['relative_path']}")
            continue
        cached = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=repo_path,
                token=token,
            )
        )
        destination.write_bytes(cached.read_bytes())
        actual_hash = sha256_file(destination)
        if actual_hash != entry["sha256"]:
            raise ValueError(f"Hash mismatch for {entry['relative_path']}")
        print(f"downloaded {entry['relative_path']}")


def collect_records(project_root: Path, include_paths: Iterable[str]) -> list[FileRecord]:
    records: list[FileRecord] = []
    for include in include_paths:
        path = safe_project_path(project_root, include)
        if not path.exists():
            raise FileNotFoundError(f"Included path does not exist: {path}")
        candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for candidate in candidates:
            if should_skip(candidate):
                continue
            relative = candidate.relative_to(project_root).as_posix()
            records.append(
                FileRecord(
                    local_path=candidate,
                    repo_path=relative,
                    size_bytes=candidate.stat().st_size,
                    sha256=sha256_file(candidate),
                )
            )
    return sorted(records, key=lambda record: record.repo_path)


def build_manifest(
    records: list[FileRecord],
    storage_mode: str,
    archives: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    archive_by_file = {}
    for archive in archives or []:
        for file_path in archive.get("files", []):
            archive_by_file[str(file_path)] = archive["repo_path"]
    return {
        "schema_version": 1,
        "storage_mode": storage_mode,
        "path_mapping": "relative_path equals project-root-relative path",
        "archives": archives or [],
        "files": [
            {
                "relative_path": record.repo_path,
                "repo_path": record.repo_path
                if storage_mode == "files"
                else archive_by_file.get(record.repo_path),
                "size_bytes": record.size_bytes,
                "sha256": record.sha256,
            }
            for record in records
        ],
    }


def upload_archives(api, token: str | None, args: argparse.Namespace, records: list[FileRecord]) -> None:
    with tempfile.TemporaryDirectory(prefix="retina_hf_archives_") as staging_dir:
        staging_root = Path(staging_dir)
        archives = create_archive_shards(staging_root, records, args.archive_max_bytes)
        manifest = build_manifest(records, storage_mode="archive", archives=archives)
        manifest_destination = staging_root / args.manifest_path
        manifest_destination.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        remote_paths = remote_repo_paths(api, args.repo_id, token)
        if not args.force:
            for archive in archives:
                if archive["repo_path"] in remote_paths:
                    archive_path = staging_root / str(archive["repo_path"])
                    if archive_path.exists():
                        archive_path.unlink()
                    print(f"remote archive exists, skipping upload: {archive['repo_path']}")
        for archive in archives:
            archive_path = staging_root / str(archive["repo_path"])
            if not archive_path.exists():
                continue
            print(f"uploading archive {archive['repo_path']}", flush=True)
            api.upload_file(
                path_or_fileobj=archive_path,
                path_in_repo=str(archive["repo_path"]),
                repo_id=args.repo_id,
                repo_type="dataset",
                token=token,
                commit_message=f"{args.commit_message}: {archive['repo_path']}",
            )
        print(f"uploading manifest {args.manifest_path}", flush=True)
        api.upload_file(
            path_or_fileobj=manifest_destination,
            path_in_repo=args.manifest_path,
            repo_id=args.repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"{args.commit_message}: manifest",
        )
        print(f"uploaded archive manifest and {len(archives)} archive shard record(s)")


def plan_archive_shards(records: list[FileRecord], max_bytes: int) -> list[list[FileRecord]]:
    shards: list[list[FileRecord]] = []
    current: list[FileRecord] = []
    current_bytes = 0
    for record in records:
        if current and current_bytes + record.size_bytes > max_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(record)
        current_bytes += record.size_bytes
    if current:
        shards.append(current)
    return shards


def create_archive_shards(
    staging_root: Path, records: list[FileRecord], max_bytes: int
) -> list[dict[str, object]]:
    archive_root = staging_root / DEFAULT_ARCHIVE_DIR
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_records: list[dict[str, object]] = []
    shards = plan_archive_shards(records, max_bytes)
    for index, shard in enumerate(shards):
        archive_repo_path = f"{DEFAULT_ARCHIVE_DIR}/retina_project_data_{index:04d}.tar"
        archive_path = staging_root / archive_repo_path
        with tarfile.open(archive_path, "w") as tar:
            for record in shard:
                tar.add(record.local_path, arcname=record.repo_path, recursive=False)
        archive_records.append(
            {
                "repo_path": archive_repo_path,
                "size_bytes": archive_path.stat().st_size,
                "sha256": sha256_file(archive_path),
                "files": [record.repo_path for record in shard],
            }
        )
        print(f"created {archive_repo_path} with {len(shard)} files")
    return archive_records


def download_archives(
    args: argparse.Namespace,
    token: str | None,
    manifest: dict[str, object],
    files: list[dict[str, object]],
    project_root: Path,
) -> None:
    from huggingface_hub import hf_hub_download

    files_by_archive: dict[str, list[dict[str, object]]] = {}
    for entry in files:
        archive_path = str(entry["repo_path"])
        files_by_archive.setdefault(archive_path, []).append(entry)

    for archive in manifest.get("archives", []):
        archive_repo_path = str(archive["repo_path"])
        entries = files_by_archive.get(archive_repo_path, [])
        if not args.force and entries and all(local_entry_current(project_root, entry) for entry in entries):
            print(f"archive current locally, skipping {archive_repo_path}")
            continue
        cached = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=archive_repo_path,
                token=token,
            )
        )
        if sha256_file(cached) != archive["sha256"]:
            raise ValueError(f"Archive hash mismatch: {archive_repo_path}")
        extract_archive(cached, project_root, entries, force=args.force)
        print(f"extracted {archive_repo_path}")


def extract_archive(
    archive_path: Path,
    project_root: Path,
    entries: list[dict[str, object]],
    *,
    force: bool,
) -> None:
    expected = {str(entry["relative_path"]): entry for entry in entries}
    with tarfile.open(archive_path, "r") as tar:
        for member in tar.getmembers():
            if member.isdir():
                continue
            if member.name not in expected:
                continue
            destination = safe_project_path(project_root, member.name)
            entry = expected[member.name]
            if not force and destination.exists() and sha256_file(destination) == entry["sha256"]:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Could not extract {member.name}")
            with destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            if sha256_file(destination) != entry["sha256"]:
                raise ValueError(f"Hash mismatch after extracting {member.name}")


def local_entry_current(project_root: Path, entry: dict[str, object]) -> bool:
    destination = safe_project_path(project_root, str(entry["relative_path"]))
    return destination.exists() and sha256_file(destination) == entry["sha256"]


def remote_repo_paths(api, repo_id: str, token: str | None) -> set[str]:
    try:
        return set(api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token))
    except Exception as exc:
        message = str(exc)
        if "404" in message or "Repository Not Found" in message:
            return set()
        raise


def skip_existing_remote(records: list[FileRecord], remote_paths: set[str]) -> list[FileRecord]:
    return [record for record in records if record.repo_path not in remote_paths]


def hf_api(token_env: str | None):
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv()

    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required. Install with `python -m pip install -r requirements.txt`."
        ) from exc

    token = resolve_token(token_env)
    return HfApi(token=token), token


def resolve_token(token_env: str | None) -> str | None:
    if token_env:
        return os.environ.get(token_env)
    for name in (
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
    ):
        token = os.environ.get(name)
        if token:
            return token
    return None


def safe_project_path(project_root: Path, relative_path: str) -> Path:
    path = (project_root / relative_path).resolve()
    if project_root not in path.parents and path != project_root:
        raise ValueError(f"Path escapes project root: {relative_path}")
    return path


def should_skip(path: Path) -> bool:
    return path.name in EXCLUDED_NAMES or any(part == "__pycache__" for part in path.parts)


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
