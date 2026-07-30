# Phase 5 production backend image: FastAPI + Uvicorn serving the Phase 3A/3B
# API (drone_sim.api.app:app). Deliberately excludes the `viz` extra
# (matplotlib/the local debug viewer) -- that is a local-development tool,
# never part of the deployed API surface. Multi-stage so the runtime image
# never carries pip's build cache or a compiler toolchain.
#
# Build:  docker build -t drone-sim-backend .
# Run:    docker run --rm -p 8000:8000 drone-sim-backend
# Health: GET http://localhost:8000/health  (liveness)
#         GET http://localhost:8000/ready   (readiness)

FROM python:3.12-slim AS builder

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Deterministic dependency installation: pyproject.toml pins numpy>=1.24 and
# fastapi>=0.100/uvicorn>=0.23 for the `api` extra; no unpinned "latest"
# dependencies are pulled in beyond what pip's own resolver locks for those
# ranges at build time.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir ".[api]"

FROM python:3.12-slim AS runtime

# Non-root execution.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

USER appuser
EXPOSE 8000

# Liveness probe hits /health (always ok once the process can respond at
# all); readiness is a separate concern exposed at /ready -- see
# src/drone_sim/api/monitoring.py.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=2)" || exit 1

# Shell form so $HOST/$PORT are honored -- e.g. `docker run -e PORT=9000 ...`.
# Uvicorn's own SIGTERM handling gives graceful shutdown (finishes in-flight
# requests, then exits) without any extra wrapper here.
CMD uvicorn drone_sim.api.app:app --host $HOST --port $PORT
