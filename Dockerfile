# syntax=docker/dockerfile:1

FROM rust:1-slim-bookworm AS builder

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release --locked

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --uid 10001 app

WORKDIR /app

COPY --from=builder /app/target/release/scut-ai-proxy /usr/local/bin/scut-ai-proxy

ENV BIND_ADDR=0.0.0.0:3000 \
    CHAT3_BASE_URL=https://chat3.scut.edu.cn/api \
    REQUEST_TIMEOUT_SECS=120 \
    PLANNER_REPAIR_ATTEMPTS=1

EXPOSE 3000

USER app

ENTRYPOINT ["scut-ai-proxy"]
