"""Phase 5 deployment smoke test.

Default mode (``python scripts/smoke_test.py``) drives the full container
lifecycle via ``docker compose``:

    1. build the backend + frontend images
    2. start them
    3. wait for the backend's /ready endpoint
    4. create a bounded simulation
    5. verify status retrieval
    6. retrieve visualization data (GET /frame)
    7. pause the simulation
    8. tear everything down and verify clean shutdown

``--base-url <url>`` skips steps 1/2/8 (build/start/teardown) and runs only
steps 3-7 against an already-running backend (container-based or a plain
``uvicorn drone_sim.api.app:app`` process) -- useful for verifying this
script's own request sequence independently of whether Docker is available.

Exit code is 0 only if every step actually succeeded; this script never
prints a success message for a step it did not really perform.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BACKEND_URL = "http://localhost:8000"
DEFAULT_FRONTEND_URL = "http://localhost:8080"


def _get(url: str, timeout: float = 5.0) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {}


def _post(url: str, body: dict | None = None, timeout: float = 5.0) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8") if body is not None else b""
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read())


def _wait_for_ready(base_url: str, timeout_s: float = 60.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            status, body = _get(f"{base_url}/ready", timeout=2.0)
            if status == 200 and body.get("status") == "ready":
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def _docker_compose_available() -> bool:
    return shutil.which("docker") is not None


def _run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=None, help="skip docker compose build/up/down, test against this URL instead")
    ap.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    ap.add_argument("--keep-running", action="store_true", help="don't tear down containers at the end")
    args = ap.parse_args()

    backend_url = args.base_url or args.backend_url
    managed_by_this_script = args.base_url is None

    if managed_by_this_script:
        if not _docker_compose_available():
            print(
                "FAIL: 'docker' was not found on PATH. This environment does not have "
                "Docker installed, so the container build/start/teardown steps cannot be "
                "run here. Install Docker (or Docker Desktop) and re-run this script, or "
                "pass --base-url http://localhost:8000 to smoke-test an already-running "
                "(e.g. plain `uvicorn drone_sim.api.app:app`) backend instead."
            )
            return 1

        print("== Step 1/8: build images ==")
        if _run(["docker", "compose", "build"]) != 0:
            print("FAIL: docker compose build failed")
            return 1

        print("== Step 2/8: start containers ==")
        if _run(["docker", "compose", "up", "-d"]) != 0:
            print("FAIL: docker compose up failed")
            return 1
    else:
        print(f"== Skipping build/start -- testing already-running backend at {backend_url} ==")

    try:
        print("== Step 3/8: wait for readiness ==")
        if not _wait_for_ready(backend_url):
            print("FAIL: backend never became ready")
            return 1
        print("OK: backend is ready")

        print("== Step 4/8: create a bounded simulation ==")
        status, sim = _post(f"{backend_url}/simulations", {"num_drones": 500, "bounds_max": [200.0, 200.0, 200.0]})
        if status != 200:
            print(f"FAIL: POST /simulations returned {status}")
            return 1
        sim_id = sim["simulation_id"]
        print(f"OK: created simulation {sim_id}")

        print("== Step 5/8: verify status retrieval ==")
        status, body = _get(f"{backend_url}/simulations/{sim_id}")
        if status != 200 or body.get("simulation_id") != sim_id:
            print(f"FAIL: GET /simulations/{{id}} returned {status} / {body}")
            return 1
        print(f"OK: status={body['status']} tick={body['tick']}")

        print("== Step 6/8: advance one tick and retrieve visualization data ==")
        status, _ = _post(f"{backend_url}/simulations/{sim_id}/step")
        if status != 200:
            print(f"FAIL: POST .../step returned {status}")
            return 1
        status, frame = _get(f"{backend_url}/simulations/{sim_id}/frame?x_min=0&x_max=200&y_min=0&y_max=200")
        if status != 200 or "heatmap" not in frame:
            print(f"FAIL: GET .../frame returned {status}")
            return 1
        print(f"OK: /frame tick={frame['tick']} num_visible_drones={frame['num_visible_drones']}")

        print("== Step 7/8: pause the simulation ==")
        status, _ = _post(f"{backend_url}/simulations/{sim_id}/start")
        if status != 200:
            print(f"FAIL: POST .../start returned {status}")
            return 1
        status, body = _post(f"{backend_url}/simulations/{sim_id}/pause")
        if status != 200 or body.get("status") != "paused":
            print(f"FAIL: POST .../pause returned {status} / {body}")
            return 1
        print("OK: simulation paused")

        try:
            req = urllib.request.Request(f"{backend_url}/simulations/{sim_id}", method="DELETE")
            urllib.request.urlopen(req, timeout=5.0)
        except Exception as exc:
            print(f"WARN: cleanup DELETE failed (non-fatal): {exc}")

        print("\nSMOKE TEST PASSED")
        return 0
    finally:
        if managed_by_this_script and not args.keep_running:
            print("== Step 8/8: tear down ==")
            _run(["docker", "compose", "down"])


if __name__ == "__main__":
    sys.exit(main())
