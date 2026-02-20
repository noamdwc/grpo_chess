"""MCP tool implementations for Lightning.ai Studio."""
import json
from typing import Optional

from .studio import get_studio, get_jobs_plugin


def _fmt(obj) -> str:
    return json.dumps(obj, default=str)


async def submit_job(command: str, name: str, machine: str = "CPU-4") -> str:
    try:
        from lightning_sdk import Machine
        studio = get_studio()
        jobs = get_jobs_plugin(studio)
        job = jobs.run(command=command, name=name, machine=Machine[machine])
        return _fmt({"job_name": name, "status": str(job.status), "machine": machine})
    except Exception as e:
        return _fmt({"error": str(e)})


async def list_jobs(limit: int = 10, state: Optional[str] = None) -> str:
    try:
        studio = get_studio()
        jobs = get_jobs_plugin(studio)
        all_jobs = jobs.list_jobs()
        result = []
        for job in all_jobs:
            status = str(job.status)
            if state and status.lower() != state.lower():
                continue
            result.append({
                "name": job.name,
                "status": status,
                "machine": str(getattr(job, "machine", "")),
                "created_at": str(getattr(job, "created_at", "")),
            })
            if len(result) >= limit:
                break
        return _fmt(result)
    except Exception as e:
        return _fmt({"error": str(e)})


async def cancel_job(job_name: str) -> str:
    try:
        studio = get_studio()
        jobs = get_jobs_plugin(studio)
        jobs.stop_job(job_name)
        return _fmt({"job_name": job_name, "cancelled": True})
    except Exception as e:
        return _fmt({"error": str(e)})


async def get_logs(job_name: str) -> str:
    try:
        studio = get_studio()
        jobs = get_jobs_plugin(studio)
        all_jobs = jobs.list_jobs()
        job = next((j for j in all_jobs if j.name == job_name), None)
        if job is None:
            return _fmt({"error": f"Job '{job_name}' not found"})
        logs = job.logs()
        return _fmt({"job_name": job_name, "logs": logs})
    except Exception as e:
        return _fmt({"error": str(e)})


async def get_credits() -> str:
    try:
        from lightning_sdk.lightning_cloud.rest_client import LightningClient
        client = LightningClient()
        resp = client.billing_service_get_user_balance()
        return _fmt({
            "balance": resp.balance,
            "total_spent": resp.total_spent,
            "account_id": resp.account_id,
        })
    except Exception as e:
        return _fmt({"error": str(e)})
