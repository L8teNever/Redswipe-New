FROM rust:1-slim-bookworm AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
RUN mkdir src && echo "fn main(){}" > src/main.rs && cargo build --release && rm -rf src
COPY src ./src
RUN touch src/main.rs && cargo build --release

FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -r -u 10001 app
COPY --from=builder /app/target/release/reddit-swipe /usr/local/bin/reddit-swipe
COPY static /app/static
RUN mkdir /vpn && chown app /vpn
WORKDIR /app
ENV PORT=8080 STATIC_DIR=/app/static VPN_DIR=/vpn
EXPOSE 8080
USER app
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl -fsS http://localhost:8080/healthz || exit 1
CMD ["reddit-swipe"]
