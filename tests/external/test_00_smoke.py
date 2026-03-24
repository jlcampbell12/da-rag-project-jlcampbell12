"""External smoke tests: hit the running API server using the requests library.

Requires the server to be running (see tests/external/conftest.py for details).
Run with: pytest tests/external/
"""

import requests


def test_health_returns_200(base_url):
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200


def test_health_content_type_is_json(base_url):
    response = requests.get(f"{base_url}/health")
    assert "application/json" in response.headers.get("content-type", "")


def test_health_status_is_ok(base_url):
    response = requests.get(f"{base_url}/health")
    body = response.json()
    assert body["status"] == "ok"


def test_health_has_service_field(base_url):
    response = requests.get(f"{base_url}/health")
    assert "service" in response.json()


def test_health_has_version_field(base_url):
    response = requests.get(f"{base_url}/health")
    assert "version" in response.json()


def test_unknown_route_returns_404(base_url):
    response = requests.get(f"{base_url}/nonexistent-route")
    assert response.status_code == 404
