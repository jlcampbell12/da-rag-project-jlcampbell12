"""External test configuration.

Tests in this directory call the live API server using the requests library.
The server must be running before these tests are executed.

Start the server:
    uv run uvicorn main:app --app-dir src --reload

Override the base URL via environment variable:
    $env:API_BASE_URL = "http://localhost:8000"
    pytest tests/external/
"""

import os

import pytest
import requests

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL


@pytest.fixture(scope="session", autouse=True)
def live_server(base_url: str):
    """Skip the entire session if the API server is not reachable."""
    try:
        requests.get(f"{base_url}/health", timeout=3)
    except requests.ConnectionError:
        pytest.skip(
            f"API server not reachable at {base_url}. "
            "Start it with: uv run uvicorn main:app --app-dir src --reload"
        )
    return base_url
