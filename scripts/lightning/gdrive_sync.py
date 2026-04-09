#!/usr/bin/env python3
"""Google Drive sync helpers for Lightning jobs.

Auth is loaded from one of:
- Service account:
  - GDRIVE_SERVICE_ACCOUNT_FILE (path to json)
  - GDRIVE_SERVICE_ACCOUNT_JSON (raw json)
  - GDRIVE_SERVICE_ACCOUNT_JSON_B64 (base64-encoded json)
- OAuth user creds (recommended for personal Google One / My Drive):
  - GDRIVE_OAUTH_CLIENT_ID
  - GDRIVE_OAUTH_CLIENT_SECRET
  - GDRIVE_OAUTH_REFRESH_TOKEN
  - optional GDRIVE_OAUTH_TOKEN_URI (default oauth2 token endpoint)
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials as UserCredentials
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
except ImportError as exc:  # pragma: no cover - runtime dependency in Lightning jobs
    raise SystemExit(
        "Missing Google Drive dependencies. Install with: "
        "python -m pip install google-api-python-client google-auth"
    ) from exc


FOLDER_MIME = "application/vnd.google-apps.folder"
SCOPES = ["https://www.googleapis.com/auth/drive"]


def _escape_q(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _load_service_account_info() -> dict[str, Any]:
    raw_path = os.environ.get("GDRIVE_SERVICE_ACCOUNT_FILE")
    if raw_path:
        return json.loads(Path(raw_path).read_text(encoding="utf-8"))

    raw_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON")
    if raw_json:
        return json.loads(raw_json)

    raw_b64 = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON_B64")
    if raw_b64:
        decoded = base64.b64decode(raw_b64).decode("utf-8")
        return json.loads(decoded)

    raise RuntimeError(
        "No Google Drive service-account credential found. "
        "Set GDRIVE_SERVICE_ACCOUNT_FILE or GDRIVE_SERVICE_ACCOUNT_JSON or "
        "GDRIVE_SERVICE_ACCOUNT_JSON_B64."
    )


def _load_oauth_user_credentials() -> UserCredentials | None:
    client_id = os.environ.get("GDRIVE_OAUTH_CLIENT_ID")
    client_secret = os.environ.get("GDRIVE_OAUTH_CLIENT_SECRET")
    refresh_token = os.environ.get("GDRIVE_OAUTH_REFRESH_TOKEN")
    if not (client_id and client_secret and refresh_token):
        return None

    token_uri = os.environ.get("GDRIVE_OAUTH_TOKEN_URI", "https://oauth2.googleapis.com/token")
    creds = UserCredentials(
        token=None,
        refresh_token=refresh_token,
        token_uri=token_uri,
        client_id=client_id,
        client_secret=client_secret,
        scopes=SCOPES,
    )
    creds.refresh(Request())
    return creds


def _load_credentials():
    # Prefer OAuth user credentials when provided (works with personal Google One drives).
    oauth_creds = _load_oauth_user_credentials()
    if oauth_creds is not None:
        return oauth_creds

    # Fallback to service account credentials.
    return service_account.Credentials.from_service_account_info(
        _load_service_account_info(),
        scopes=SCOPES,
    )


def _build_drive():
    return build("drive", "v3", credentials=_load_credentials(), cache_discovery=False)


def _list_children(drive, parent_id: str, *, name: str | None = None, mime_type: str | None = None) -> list[dict]:
    q_parts = [f"'{_escape_q(parent_id)}' in parents", "trashed=false"]
    if name is not None:
        q_parts.append(f"name='{_escape_q(name)}'")
    if mime_type is not None:
        q_parts.append(f"mimeType='{_escape_q(mime_type)}'")

    files: list[dict] = []
    page_token: str | None = None
    while True:
        resp = drive.files().list(
            q=" and ".join(q_parts),
            spaces="drive",
            fields="nextPageToken, files(id, name, mimeType, size)",
            pageToken=page_token,
            pageSize=1000,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def _find_child(drive, parent_id: str, name: str, *, mime_type: str | None = None) -> dict | None:
    found = _list_children(drive, parent_id, name=name, mime_type=mime_type)
    return found[0] if found else None


def _ensure_folder(drive, parent_id: str, name: str) -> str:
    existing = _find_child(drive, parent_id, name, mime_type=FOLDER_MIME)
    if existing:
        return existing["id"]

    created = drive.files().create(
        body={
            "name": name,
            "mimeType": FOLDER_MIME,
            "parents": [parent_id],
        },
        fields="id",
    ).execute()
    return created["id"]


def _resolve_folder(drive, root_folder_id: str, rel_dir: str, *, create: bool) -> str | None:
    parts = [p for p in rel_dir.split("/") if p and p != "."]
    current = root_folder_id
    for part in parts:
        existing = _find_child(drive, current, part, mime_type=FOLDER_MIME)
        if existing:
            current = existing["id"]
            continue
        if not create:
            return None
        current = _ensure_folder(drive, current, part)
    return current


def _resolve_remote_file(drive, root_folder_id: str, remote_path: str) -> dict | None:
    parent_path, _, file_name = remote_path.rpartition("/")
    parent_id = _resolve_folder(drive, root_folder_id, parent_path, create=False)
    if not parent_id:
        return None
    return _find_child(drive, parent_id, file_name, mime_type=None)


def _download_file_by_id(drive, file_id: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = local_path.with_suffix(local_path.suffix + ".part")
    request = drive.files().get_media(fileId=file_id)
    with temp_path.open("wb") as handle:
        downloader = MediaIoBaseDownload(handle, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    os.replace(temp_path, local_path)


def cmd_download_file(args: argparse.Namespace) -> int:
    drive = _build_drive()
    remote = _resolve_remote_file(drive, args.folder_id, args.remote_path)
    if not remote:
        if args.optional:
            print(f"Remote file missing (optional): {args.remote_path}")
            return 0
        print(f"Remote file not found: {args.remote_path}", file=sys.stderr)
        return 2
    _download_file_by_id(drive, remote["id"], Path(args.local_path))
    print(f"Downloaded file: {args.remote_path} -> {args.local_path}")
    return 0


def cmd_upload_file(args: argparse.Namespace) -> int:
    local_path = Path(args.local_path)
    if not local_path.is_file():
        print(f"Local file not found: {local_path}", file=sys.stderr)
        return 2

    drive = _build_drive()
    parent_path, _, file_name = args.remote_path.rpartition("/")
    parent_id = _resolve_folder(drive, args.folder_id, parent_path, create=True)
    assert parent_id is not None
    existing = _find_child(drive, parent_id, file_name, mime_type=None)

    local_size = local_path.stat().st_size
    if existing and not args.force:
        remote_size = existing.get("size")
        if remote_size is not None and int(remote_size) == local_size:
            print(f"Skipped unchanged file: {args.remote_path}")
            return 0

    media = MediaFileUpload(str(local_path), resumable=True)
    if existing:
        drive.files().update(fileId=existing["id"], media_body=media, fields="id").execute()
        print(f"Updated file: {local_path} -> {args.remote_path}")
    else:
        drive.files().create(
            body={"name": file_name, "parents": [parent_id]},
            media_body=media,
            fields="id",
        ).execute()
        print(f"Uploaded file: {local_path} -> {args.remote_path}")
    return 0


def cmd_download_glob(args: argparse.Namespace) -> int:
    drive = _build_drive()
    remote_dir_id = _resolve_folder(drive, args.folder_id, args.remote_dir, create=False)
    if not remote_dir_id:
        if args.optional:
            print(f"Remote dir missing (optional): {args.remote_dir}")
            return 0
        print(f"Remote dir not found: {args.remote_dir}", file=sys.stderr)
        return 2

    remote_files = [f for f in _list_children(drive, remote_dir_id) if f.get("mimeType") != FOLDER_MIME]
    matched = sorted(
        (f for f in remote_files if fnmatch.fnmatch(f["name"], args.glob)),
        key=lambda x: x["name"],
    )
    if args.max_files and args.max_files > 0:
        matched = matched[: args.max_files]

    if not matched:
        if args.optional:
            print(f"No remote matches (optional): {args.remote_dir}/{args.glob}")
            return 0
        print(f"No remote matches: {args.remote_dir}/{args.glob}", file=sys.stderr)
        return 2

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    for meta in matched:
        _download_file_by_id(drive, meta["id"], local_dir / meta["name"])
        print(f"Downloaded: {args.remote_dir}/{meta['name']}")
    return 0


def cmd_upload_glob(args: argparse.Namespace) -> int:
    local_dir = Path(args.local_dir)
    if not local_dir.is_dir():
        if args.optional:
            print(f"Local dir missing (optional): {local_dir}")
            return 0
        print(f"Local dir not found: {local_dir}", file=sys.stderr)
        return 2

    local_files = sorted(p for p in local_dir.iterdir() if p.is_file() and fnmatch.fnmatch(p.name, args.glob))
    if args.max_files and args.max_files > 0:
        local_files = local_files[: args.max_files]
    if not local_files:
        if args.optional:
            print(f"No local matches (optional): {local_dir}/{args.glob}")
            return 0
        print(f"No local matches: {local_dir}/{args.glob}", file=sys.stderr)
        return 2

    status = 0
    for path in local_files:
        remote_path = f"{args.remote_dir.rstrip('/')}/{path.name}"
        sub_args = argparse.Namespace(
            folder_id=args.folder_id,
            local_path=str(path),
            remote_path=remote_path,
            force=args.force,
        )
        code = cmd_upload_file(sub_args)
        if code != 0:
            status = code
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Google Drive sync helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    download_file = sub.add_parser("download-file")
    download_file.add_argument("--folder-id", required=True)
    download_file.add_argument("--remote-path", required=True)
    download_file.add_argument("--local-path", required=True)
    download_file.add_argument("--optional", action="store_true")
    download_file.set_defaults(func=cmd_download_file)

    upload_file = sub.add_parser("upload-file")
    upload_file.add_argument("--folder-id", required=True)
    upload_file.add_argument("--local-path", required=True)
    upload_file.add_argument("--remote-path", required=True)
    upload_file.add_argument("--force", action="store_true")
    upload_file.set_defaults(func=cmd_upload_file)

    download_glob = sub.add_parser("download-glob")
    download_glob.add_argument("--folder-id", required=True)
    download_glob.add_argument("--remote-dir", required=True)
    download_glob.add_argument("--local-dir", required=True)
    download_glob.add_argument("--glob", required=True)
    download_glob.add_argument("--max-files", type=int, default=0)
    download_glob.add_argument("--optional", action="store_true")
    download_glob.set_defaults(func=cmd_download_glob)

    upload_glob = sub.add_parser("upload-glob")
    upload_glob.add_argument("--folder-id", required=True)
    upload_glob.add_argument("--local-dir", required=True)
    upload_glob.add_argument("--remote-dir", required=True)
    upload_glob.add_argument("--glob", required=True)
    upload_glob.add_argument("--max-files", type=int, default=0)
    upload_glob.add_argument("--optional", action="store_true")
    upload_glob.add_argument("--force", action="store_true")
    upload_glob.set_defaults(func=cmd_upload_glob)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"Google Drive sync failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
