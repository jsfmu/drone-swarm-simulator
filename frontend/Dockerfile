# Phase 5 production frontend image: Vite production build of the React
# dashboard, served as static files. `VITE_API_BASE_URL` must be supplied as
# a BUILD ARG (Vite substitutes import.meta.env.* at build time, not runtime
# -- see SimulationDashboard.jsx), not a container-runtime env var.
#
# Build:  docker build -t drone-sim-frontend --build-arg VITE_API_BASE_URL=http://localhost:8000 ./frontend
# Run:    docker run --rm -p 8080:8080 drone-sim-frontend

FROM node:20-alpine AS builder

WORKDIR /build
COPY package.json package-lock.json* ./
RUN npm ci

COPY . .
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

# nginx-unprivileged: an official-pattern image that already runs nginx as a
# non-root user on an unprivileged port (8080), so no extra USER/permission
# wrangling is needed to satisfy "non-root container execution where
# practical".
FROM nginxinc/nginx-unprivileged:1.27-alpine AS runtime

COPY --from=builder /build/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -q -O- http://127.0.0.1:8080/ || exit 1
