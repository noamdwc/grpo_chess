import asyncio
import json
from types import SimpleNamespace

from mcp_lightning_server import tools


class _FakeJob:
    def __init__(self, name: str, status: str, machine: str = "CPU", logs: str = "ok"):
        self.name = name
        self.status = status
        self.machine = machine
        self._logs = logs
        self.teamspace = SimpleNamespace(id="project-fallback")
        self._internal_job = SimpleNamespace(
            _job=SimpleNamespace(
                id=f"{name}-id",
                project_id="project-123",
                spec=SimpleNamespace(cloudspace_id="cloudspace-xyz"),
            )
        )

    @property
    def logs(self):
        return self._logs

    def stop(self):
        self.status = "Stopped"


def _parse(result: str):
    return json.loads(result)


def test_resolve_machine_cpu_dash_4():
    machine = tools._resolve_machine("CPU-4")
    assert machine.name == "CPU_X_4"


def test_submit_job_resolves_machine(monkeypatch):
    captured = {}

    class _FakeJobsPlugin:
        def run(self, command, name, machine):
            captured["machine"] = machine
            return _FakeJob(name=name, status="Pending")

    monkeypatch.setattr(tools, "get_studio", lambda: object())
    monkeypatch.setattr(tools, "get_jobs_plugin", lambda _: _FakeJobsPlugin())

    result = _parse(asyncio.run(tools.submit_job(command="echo hi", name="job1", machine="CPU-4")))
    assert result["job_name"] == "job1"
    assert captured["machine"].name == "CPU_X_4"


def test_list_jobs_state_alias(monkeypatch):
    jobs = [
        _FakeJob(name="a", status="Completed"),
        _FakeJob(name="b", status="Pending"),
    ]
    studio = SimpleNamespace(teamspace=SimpleNamespace(jobs=jobs))
    monkeypatch.setattr(tools, "get_studio", lambda: studio)

    result = _parse(asyncio.run(tools.list_jobs(state="succeeded")))
    assert [j["name"] for j in result] == ["a"]


def test_cancel_job_not_found(monkeypatch):
    studio = SimpleNamespace(teamspace=SimpleNamespace(jobs=[]))
    monkeypatch.setattr(tools, "get_studio", lambda: studio)

    result = _parse(asyncio.run(tools.cancel_job(job_name="missing")))
    assert "not found" in result["error"].lower()


def test_get_logs_404_is_friendly(monkeypatch):
    class _MissingLogsJob(_FakeJob):
        @property
        def logs(self):
            raise RuntimeError("HTTP Error 404: Not Found")

    studio = SimpleNamespace(teamspace=SimpleNamespace(jobs=[_MissingLogsJob("x", "Stopped")]))
    monkeypatch.setattr(tools, "get_studio", lambda: studio)

    result = _parse(asyncio.run(tools.get_logs(job_name="x")))
    assert "not available yet" in result["error"].lower()


def test_get_live_logs_running_uses_live_api(monkeypatch):
    class _FakePage:
        def __init__(self, page_number: str, total_lines: str, url: str):
            self.page_number = page_number
            self.total_lines = total_lines
            self.url = url

    class _FakeLightningClient:
        def jobs_service_get_job_logs(self, project_id, id, **kwargs):
            assert project_id == "project-123"
            assert id == "run-id"
            assert kwargs["cloudspace_id"] == "cloudspace-xyz"
            return SimpleNamespace(
                follow_url="https://follow",
                pages=[
                    _FakePage("1", "2", "https://logs/page-1"),
                    _FakePage("2", "2", "https://logs/page-2"),
                ],
            )

    class _RunningJob(_FakeJob):
        def __init__(self):
            super().__init__(name="run", status="Running", logs="unused")
            self._internal_job = SimpleNamespace(
                _job=SimpleNamespace(
                    id="run-id",
                    project_id="project-123",
                    spec=SimpleNamespace(cloudspace_id="cloudspace-xyz"),
                )
            )

    studio = SimpleNamespace(teamspace=SimpleNamespace(jobs=[_RunningJob()]))
    monkeypatch.setattr(tools, "get_studio", lambda: studio)
    monkeypatch.setattr(tools, "LightningClient", lambda: _FakeLightningClient())
    monkeypatch.setattr(
        tools,
        "_fetch_url_text",
        lambda url: "[2026] line-1\n[2026] line-2\n" if "page-1" in url else "[2026] line-3\n[2026] line-4\n",
    )

    result = _parse(asyncio.run(tools.get_live_logs(job_name="run", tail_lines=2)))
    assert result["ok"] is True
    assert result["source"] == "live_api"
    assert result["tail_lines_requested"] == 2
    assert result["job_id"] == "run-id"
    assert "line-4" in result["logs"]
    assert "line-3" in result["logs"]


def test_get_live_logs_terminal_fallback(monkeypatch):
    studio = SimpleNamespace(teamspace=SimpleNamespace(jobs=[_FakeJob("done", "Completed", logs="a\nb\nc\n")]))
    monkeypatch.setattr(tools, "get_studio", lambda: studio)

    result = _parse(asyncio.run(tools.get_live_logs(job_name="done", tail_lines=2)))
    assert result["ok"] is True
    assert result["source"] == "final_logs_fallback"
    assert result["tail_lines_requested"] == 2
    assert result["tail_lines_returned"] == 2
    assert result["logs"] == "b\nc"
