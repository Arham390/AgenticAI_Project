"""Unit tests — Phase 4: Web Interface (FastAPI endpoints)."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from fastapi.testclient import TestClient
    _HAS_TESTCLIENT = True
except ImportError:
    _HAS_TESTCLIENT = False

import pytest


@pytest.fixture(scope="module")
def client():
    if not _HAS_TESTCLIENT:
        pytest.skip("fastapi[testclient] not installed")
    from web.app import app
    return TestClient(app)


class TestAPIEndpoints:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code in (200, 307)

    def test_docs_available(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_run_endpoint_requires_prompt(self, client):
        resp = client.post("/api/run", json={})
        assert resp.status_code == 400

    def test_run_endpoint_returns_job_id(self, client):
        resp = client.post("/api/run", json={"prompt": "Test prompt for unit test."})
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert isinstance(data["job_id"], str)

    def test_job_poll_unknown_returns_404(self, client):
        resp = client.get("/api/job/nonexistent-job-id")
        assert resp.status_code == 404

    def test_job_poll_known_job(self, client):
        resp = client.post("/api/run", json={"prompt": "Quick test."})
        job_id = resp.json()["job_id"]
        resp2 = client.get(f"/api/job/{job_id}")
        assert resp2.status_code == 200
        data = resp2.json()
        assert "status" in data

    def test_outputs_endpoint_returns_dict(self, client):
        resp = client.get("/api/outputs")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_history_endpoint(self, client):
        resp = client.get("/api/history")
        assert resp.status_code == 200
        data = resp.json()
        assert "versions" in data
        assert isinstance(data["versions"], list)

    def test_current_version_endpoint(self, client):
        resp = client.get("/api/version/current")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data

    def test_revert_nonexistent_version(self, client):
        resp = client.post("/api/revert/99999")
        assert resp.status_code == 404

    def test_edit_endpoint_requires_query(self, client):
        resp = client.post("/api/edit", json={})
        assert resp.status_code == 400

    def test_edit_endpoint_returns_job_id(self, client):
        resp = client.post("/api/edit", json={"query": "Make the scene darker."})
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data

    def test_file_serving_missing_file(self, client):
        resp = client.get("/api/file/nonexistent/file.txt")
        assert resp.status_code == 404

    def test_file_serving_path_traversal_blocked(self, client):
        resp = client.get("/api/file/../../../etc/passwd")
        assert resp.status_code in (403, 404, 422)

    def test_phase_rerun_without_pipeline_first(self, client):
        resp = client.post("/api/phase/voice_synth", json={})
        # Either 400 (no manifest yet) or 200 (job queued if manifest exists)
        assert resp.status_code in (200, 400)
