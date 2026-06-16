# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12-slim@sha256:d764629ce0ddd8c71fd371e9901efb324a95789d2315a47db7e4d27e78f1b0e9
ARG VARIANT=dev
ARG SOURCE_URL=https://github.com/AlysisAi/Sylliptor
ARG GIT_SHA=dev
ARG VERSION=dev

FROM ${PYTHON_IMAGE} AS base
SHELL ["/bin/sh", "-euxc"]

ARG TARGETPLATFORM
ARG SOURCE_URL
ARG GIT_SHA
ARG VERSION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

LABEL org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.title="Sylliptor Sandbox" \
      org.opencontainers.image.description="Minimal Sylliptor shell sandbox runtime image." \
      org.opencontainers.image.licenses="Apache-2.0"

RUN apt-get update; \
    packages="bash ca-certificates curl git ripgrep build-essential python3 python3-venv tini"; \
    case "${TARGETPLATFORM:-linux/amd64}" in \
        linux/*) packages="${packages} bubblewrap" ;; \
        *) echo "Skipping bubblewrap for TARGETPLATFORM=${TARGETPLATFORM:-unknown}" ;; \
    esac; \
    pinned=""; \
    for package in ${packages}; do \
        version="$(apt-cache policy "${package}" | awk '/Candidate:/ {print $2; exit}')"; \
        if [ -z "${version}" ] || [ "${version}" = "(none)" ]; then \
            echo "No apt candidate found for ${package}" >&2; \
            exit 1; \
        fi; \
        echo "APT pin: ${package}=${version}"; \
        pinned="${pinned} ${package}=${version}"; \
    done; \
    apt-get install -y --no-install-recommends ${pinned}; \
    apt-mark manual ${packages}; \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 sylliptor; \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash sylliptor; \
    install -d -m 0755 -o 1000 -g 1000 /workspace

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD git --version && rg --version
CMD ["sh", "-lc", "echo sylliptor-sandbox ready"]
WORKDIR /workspace
USER sylliptor

FROM base AS dev
SHELL ["/bin/sh", "-euxc"]

USER root

# Node 24 is the active LTS line as of 2026-05. Override NODE_MAJOR when building
# a compatibility image for an older runtime.
ARG NODE_MAJOR=24
# Keep npm ahead of the NodeSource package's bundled npm when scanner-actionable
# transitive npm dependencies receive security fixes.
ARG NPM_VERSION=11.13.0
ARG TARGETARCH
ARG GO_VERSION=1.26.2
ARG GO_LINUX_AMD64_SHA256=990e6b4bbba816dc3ee129eaeaf4b42f17c2800b88a2166c265ac1a200262282
ARG GO_LINUX_ARM64_SHA256=c958a1fe1b361391db163a485e21f5f228142d6f8b584f6bef89b26f66dc5b23

ENV RUSTUP_HOME=/home/sylliptor/.rustup \
    CARGO_HOME=/home/sylliptor/.cargo \
    PATH=/home/sylliptor/.cargo/bin:/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update; \
    packages="gnupg pkg-config"; \
    pinned=""; \
    for package in ${packages}; do \
        version="$(apt-cache policy "${package}" | awk '/Candidate:/ {print $2; exit}')"; \
        if [ -z "${version}" ] || [ "${version}" = "(none)" ]; then \
            echo "No apt candidate found for ${package}" >&2; \
            exit 1; \
        fi; \
        echo "APT pin: ${package}=${version}"; \
        pinned="${pinned} ${package}=${version}"; \
    done; \
    apt-get install -y --no-install-recommends ${pinned}; \
    apt-mark manual ${packages}; \
    rm -rf /var/lib/apt/lists/*

RUN install -d -m 0755 /etc/apt/keyrings; \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor --yes -o /etc/apt/keyrings/nodesource.gpg; \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list; \
    apt-get update; \
    package=nodejs; \
    version="$(apt-cache policy "${package}" | awk '/Candidate:/ {print $2; exit}')"; \
    if [ -z "${version}" ] || [ "${version}" = "(none)" ]; then \
        echo "No apt candidate found for ${package}" >&2; \
        exit 1; \
    fi; \
    echo "APT pin: ${package}=${version}"; \
    apt-get install -y --no-install-recommends "${package}=${version}"; \
    npm install -g "npm@${NPM_VERSION}"; \
    npm cache clean --force; \
    apt-mark manual "${package}"; \
    rm -rf /var/lib/apt/lists/*

RUN case "${TARGETARCH:-amd64}" in \
        amd64) go_arch=amd64; go_sha="${GO_LINUX_AMD64_SHA256}" ;; \
        arm64) go_arch=arm64; go_sha="${GO_LINUX_ARM64_SHA256}" ;; \
        *) echo "Unsupported Go TARGETARCH=${TARGETARCH:-unknown}" >&2; exit 1 ;; \
    esac; \
    curl -fsSLo /tmp/go.tgz "https://go.dev/dl/go${GO_VERSION}.linux-${go_arch}.tar.gz"; \
    echo "${go_sha}  /tmp/go.tgz" | sha256sum -c -; \
    rm -rf /usr/local/go; \
    tar -C /usr/local -xzf /tmp/go.tgz; \
    rm /tmp/go.tgz; \
    go version

USER sylliptor
RUN curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path; \
    rustc -V; \
    cargo -V

USER root
RUN node -v; \
    npm -v; \
    go version; \
    pkg-config --version

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD git --version && rg --version
CMD ["sh", "-lc", "echo sylliptor-sandbox ready"]
WORKDIR /workspace
USER sylliptor

FROM dev AS server
SHELL ["/bin/sh", "-euxc"]

USER root
WORKDIR /opt/sylliptor
COPY pyproject.toml README.md /opt/sylliptor/
COPY src /opt/sylliptor/src
RUN python -m pip install --no-cache-dir /opt/sylliptor

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD git --version && rg --version
CMD ["sh", "-lc", "echo sylliptor-sandbox ready"]
WORKDIR /workspace
USER sylliptor

FROM ${VARIANT} AS final
USER sylliptor
