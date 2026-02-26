"""MCP tool implementations for Lightning.ai Studio."""
import json
import re
from typing import Optional
from urllib.request import urlopen

from lightning_sdk.lightning_cloud.rest_client import LightningClient

from .studio import get_studio, get_jobs_plugin


def _fmt(obj) -> str:
    return json.dumps(obj, default=str)


def _resolve_machine(machine: str):
    from lightning_sdk import Machine

    normalized = machine.replace("-", "_").upper()
    alias_map = {
        "CPU_2": "CPU_X_2",
        "CPU_4": "CPU_X_4",
        "CPU_8": "CPU_X_8",
        "CPU_16": "CPU_X_16",
    }
    normalized = alias_map.get(normalized, normalized)
    # Prefer named machine constants (e.g. CPU_X_4), then fall back to SDK parser.
    return getattr(Machine, normalized, Machine.from_str(machine))


def _normalize_state(state: Optional[str]) -> Optional[str]:
    if state is None:
        return None
    value = state.strip().lower()
    aliases = {
        "succeeded": "completed",
        "success": "completed",
        "cancelled": "stopped",
        "canceled": "stopped",
    }
    return aliases.get(value, value)


def _to_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _tail_text(text: str, tail_lines: int) -> str:
    lines = text.splitlines()
    if tail_lines <= 0 or len(lines) <= tail_lines:
        return text
    return "\n".join(lines[-tail_lines:])


def _strip_datetime_prefix(text: str) -> str:
    # Logs often include a leading timestamp prefix: [....] <line>.
    return re.sub(r"^\[.*?\]\s+", "", text, flags=re.MULTILINE)


def _get_job_by_name(job_name: str):
    studio = get_studio()
    return next((j for j in studio.teamspace.jobs if j.name == job_name), None)


def _safe_getattr(obj: object, attr: str):
    try:
        return getattr(obj, attr, None)
    except Exception:
        return None


def _get_job_identifiers(job) -> tuple[Optional[str], Optional[str], Optional[str]]:
    # SDK wrappers keep IDs on internal fields depending on v1/v2 job backend.
    candidates = [
        job,
        _safe_getattr(job, "_job"),
        _safe_getattr(job, "_guaranteed_job"),
        _safe_getattr(job, "_internal_job"),
    ]
    internal_job = _safe_getattr(job, "_internal_job")
    if internal_job is not None:
        candidates.extend(
            [
                _safe_getattr(internal_job, "_job"),
                _safe_getattr(internal_job, "_guaranteed_job"),
            ]
        )

    job_id: Optional[str] = None
    project_id: Optional[str] = None
    cloudspace_id: Optional[str] = None
    for candidate in candidates:
        if candidate is None:
            continue
        if job_id is None:
            value = _safe_getattr(candidate, "id")
            if value:
                job_id = str(value)
        if project_id is None:
            value = _safe_getattr(candidate, "project_id")
            if value:
                project_id = str(value)
        if cloudspace_id is None:
            spec = _safe_getattr(candidate, "spec")
            value = _safe_getattr(spec, "cloudspace_id")
            if value:
                cloudspace_id = str(value)

    if project_id is None:
        teamspace = _safe_getattr(job, "teamspace")
        value = _safe_getattr(teamspace, "id")
        if value:
            project_id = str(value)

    return job_id, project_id, cloudspace_id


def _is_terminal_status(status: str) -> bool:
    return status.strip().lower() in {
        "completed",
        "failed",
        "stopped",
        "cancelled",
        "canceled",
        "succeeded",
        "success",
    }


def _page_sort_key(page) -> tuple[int, str]:
    page_number = getattr(page, "page_number", None)
    parsed = _to_int(page_number)
    if parsed is None:
        return (10**9, str(page_number or ""))
    return (parsed, "")


def _select_pages_for_tail(pages: list[object], tail_lines: int) -> list[object]:
    if not pages:
        return []
    ordered = sorted(pages, key=_page_sort_key)
    parsed_counts = [_to_int(getattr(page, "total_lines", None)) for page in ordered]
    has_counts = any((count is not None and count > 0) for count in parsed_counts)
    if not has_counts:
        return ordered[-min(5, len(ordered)) :]

    selected_reversed = []
    lines_accumulated = 0
    for page, count in zip(reversed(ordered), reversed(parsed_counts)):
        selected_reversed.append(page)
        if count is not None and count > 0:
            lines_accumulated += count
        if lines_accumulated >= max(1, tail_lines):
            break
    return list(reversed(selected_reversed))


def _fetch_url_text(url: str) -> str:
    with urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8", errors="replace")


async def submit_job(command: str, name: str, machine: str = "CPU-4") -> str:
    try:
        studio = get_studio()
        jobs = get_jobs_plugin(studio)
        machine_obj = _resolve_machine(machine)
        job = jobs.run(command=command, name=name, machine=machine_obj)
        return _fmt({"job_name": name, "status": str(job.status), "machine": machine})
    except Exception as e:
        return _fmt({"error": str(e)})


async def list_jobs(limit: int = 10, state: Optional[str] = None) -> str:
    try:
        studio = get_studio()
        all_jobs = studio.teamspace.jobs
        state_filter = _normalize_state(state)
        result = []
        for job in all_jobs:
            status = str(job.status).lower()
            if state_filter and status != state_filter:
                continue
            result.append({
                "name": job.name,
                "status": str(job.status),
                "machine": str(getattr(job, "machine", "") or ""),
                "created_at": str(getattr(job, "created_at", "") or ""),
            })
            if len(result) >= limit:
                break
        return _fmt(result)
    except Exception as e:
        return _fmt({"error": str(e)})


async def cancel_job(job_name: str) -> str:
    try:
        job = _get_job_by_name(job_name)
        if job is None:
            return _fmt({"error": f"Job '{job_name}' not found"})
        job.stop()
        return _fmt({"job_name": job_name, "cancelled": True})
    except Exception as e:
        return _fmt({"error": str(e)})


async def get_logs(job_name: str) -> str:
    try:
        job = _get_job_by_name(job_name)
        if job is None:
            return _fmt({"error": f"Job '{job_name}' not found"})
        logs = job.logs
        return _fmt({"job_name": job_name, "logs": logs})
    except Exception as e:
        error = str(e)
        if "404" in error:
            return _fmt({"error": f"Logs for job '{job_name}' are not available yet"})
        return _fmt({"error": error})


async def get_live_logs(job_name: str, tail_lines: int = 200) -> str:
    try:
        tail_lines_value = _to_int(tail_lines)
        tail_lines = max(1, min(5000, tail_lines_value or 200))
        job = _get_job_by_name(job_name)
        if job is None:
            return _fmt({"ok": False, "error": f"Job '{job_name}' not found"})

        status = str(getattr(job, "status", "unknown")).lower()
        job_id, project_id, cloudspace_id = _get_job_identifiers(job)
        if not job_id or not project_id:
            return _fmt(
                {
                    "ok": False,
                    "job_name": job_name,
                    "state": status,
                    "error": "Could not resolve job/project IDs for live log retrieval",
                }
            )

        if _is_terminal_status(status):
            logs = job.logs
            tailed = _tail_text(logs, tail_lines)
            return _fmt(
                {
                    "ok": True,
                    "job_name": job_name,
                    "job_id": job_id,
                    "state": status,
                    "source": "final_logs_fallback",
                    "tail_lines_requested": tail_lines,
                    "tail_lines_returned": len(tailed.splitlines()),
                    "logs": tailed,
                }
            )

        client = LightningClient()
        query_kwargs = {}
        if cloudspace_id:
            query_kwargs["cloudspace_id"] = cloudspace_id
        logs_response = client.jobs_service_get_job_logs(project_id=project_id, id=job_id, **query_kwargs)
        pages = list(getattr(logs_response, "pages", None) or [])
        selected_pages = _select_pages_for_tail(pages, tail_lines)
        logs_parts: list[str] = []
        fetch_errors: list[str] = []

        for page in selected_pages:
            page_url = getattr(page, "url", None)
            if not page_url:
                continue
            try:
                logs_parts.append(_fetch_url_text(str(page_url)))
            except Exception as exc:
                fetch_errors.append(str(exc))

        combined_logs = _strip_datetime_prefix("".join(logs_parts))
        tailed_logs = _tail_text(combined_logs, tail_lines)
        result = {
            "ok": True,
            "job_name": job_name,
            "job_id": job_id,
            "state": status,
            "source": "live_api",
            "tail_lines_requested": tail_lines,
            "tail_lines_returned": len(tailed_logs.splitlines()),
            "logs": tailed_logs,
            "meta": {
                "pages_available": len(pages),
                "pages_selected": len(selected_pages),
                "follow_url": getattr(logs_response, "follow_url", None),
            },
        }
        if fetch_errors:
            result["fetch_errors"] = fetch_errors
        if not tailed_logs and not fetch_errors:
            result["meta"]["note"] = (
                "No log lines were available yet. Poll again while the job is running."
            )
        return _fmt(result)
    except Exception as e:
        return _fmt({"ok": False, "job_name": job_name, "error": str(e)})


async def get_credits() -> str:
    try:
        client = LightningClient()
        resp = client.billing_service_get_user_balance()
        return _fmt({
            "balance": resp.balance,
            "total_spent": resp.total_spent,
            "account_id": resp.account_id,
        })
    except Exception as e:
        return _fmt({"error": str(e)})
