# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12-slim@sha256:d764629ce0ddd8c71fd371e9901efb324a95789d2315a47db7e4d27e78f1b0e9
ARG VARIANT=dev
ARG SOURCE_URL=https://github.com/AlysisAi/Sylliptor
ARG GIT_SHA=dev
ARG VERSION=dev
ARG SOURCE_DATE_EPOCH=0

FROM ${PYTHON_IMAGE} AS base
SHELL ["/bin/sh", "-euxc"]

ARG TARGETPLATFORM
ARG SOURCE_URL
ARG GIT_SHA
ARG VERSION
ARG SOURCE_DATE_EPOCH
# Freeze Debian package indexes instead of resolving against a moving mirror.
ARG DEBIAN_SNAPSHOT=20260701T000000Z

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}

LABEL org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.title="Sylliptor Sandbox" \
      org.opencontainers.image.description="Minimal Sylliptor shell sandbox runtime image." \
      org.opencontainers.image.licenses="Apache-2.0"

RUN find /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) \
        -exec sed -ri \
          -e "s#https?://(deb|security)\.debian\.org/debian-security#https://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}/#g" \
          -e "s#https?://deb\.debian\.org/debian#https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}/#g" \
          {} +; \
    printf 'Acquire::Check-Valid-Until "false";\n' > /etc/apt/apt.conf.d/99snapshot; \
    apt-get update; \
    packages="bash ca-certificates curl git ripgrep build-essential python3 python3-venv tini xz-utils"; \
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

ARG TARGETARCH
# Each downloaded tool is version- and digest-pinned for both supported platforms.
ARG NODE_VERSION=24.18.1
ARG NPM_VERSION=12.0.1
ARG NPM_SHA256=5e02bea4c784df1c3bbea9e55c7d2232329e1d1920c254789833ed9e8b0a5f16
ARG NODE_LINUX_AMD64_SHA256=d6c664df3f3f61458e8c277585571328522d705166723a7c7823a9253a4d15a0
ARG NODE_LINUX_ARM64_SHA256=7201e3a09dc825bac57867c81913e2b8f0ef87d04cb9082af4cda82f6ff3d88c
ARG GO_VERSION=1.26.5
ARG GO_LINUX_AMD64_SHA256=5c2c3b16caefa1d968a94c1daca04a7ca301a496d9b086e17ad77bb81393f053
ARG GO_LINUX_ARM64_SHA256=fe4789e92b1f33358680864bbe8704289e7bb5fc207d80623c308935bd696d49
ARG RUSTUP_VERSION=1.28.2
ARG RUSTUP_LINUX_AMD64_SHA256=20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c
ARG RUSTUP_LINUX_ARM64_SHA256=e3853c5a252fca15252d07cb23a1bdd9377a8c6f3efa01531109281ae47f841c
ARG RUST_VERSION=1.91.1

ENV RUSTUP_HOME=/home/sylliptor/.rustup \
    CARGO_HOME=/home/sylliptor/.cargo \
    PATH=/home/sylliptor/.cargo/bin:/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update; \
    packages="pkg-config"; \
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

RUN case "${TARGETARCH:-amd64}" in \
        amd64) node_arch=x64; node_sha="${NODE_LINUX_AMD64_SHA256}" ;; \
        arm64) node_arch=arm64; node_sha="${NODE_LINUX_ARM64_SHA256}" ;; \
        *) echo "Unsupported Node TARGETARCH=${TARGETARCH:-unknown}" >&2; exit 1 ;; \
    esac; \
    node_archive="node-v${NODE_VERSION}-linux-${node_arch}.tar.xz"; \
    curl --proto '=https' --tlsv1.2 -fsSLo /tmp/node.tar.xz \
        "https://nodejs.org/dist/v${NODE_VERSION}/${node_archive}"; \
    echo "${node_sha}  /tmp/node.tar.xz" | sha256sum -c -; \
    tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1; \
    rm /tmp/node.tar.xz; \
    curl --proto '=https' --tlsv1.2 -fsSLo /tmp/npm.tgz \
        "https://registry.npmjs.org/npm/-/npm-${NPM_VERSION}.tgz"; \
    echo "${NPM_SHA256}  /tmp/npm.tgz" | sha256sum -c -; \
    npm --global install /tmp/npm.tgz; \
    rm /tmp/npm.tgz; \
    npm cache clean --force; \
    node --version; \
    npm --version

RUN case "${TARGETARCH:-amd64}" in \
        amd64) go_arch=amd64; go_sha="${GO_LINUX_AMD64_SHA256}" ;; \
        arm64) go_arch=arm64; go_sha="${GO_LINUX_ARM64_SHA256}" ;; \
        *) echo "Unsupported Go TARGETARCH=${TARGETARCH:-unknown}" >&2; exit 1 ;; \
    esac; \
    curl --proto '=https' --tlsv1.2 -fsSLo /tmp/go.tgz \
        "https://go.dev/dl/go${GO_VERSION}.linux-${go_arch}.tar.gz"; \
    echo "${go_sha}  /tmp/go.tgz" | sha256sum -c -; \
    rm -rf /usr/local/go; \
    tar -C /usr/local -xzf /tmp/go.tgz; \
    rm /tmp/go.tgz; \
    go version

USER sylliptor
RUN case "${TARGETARCH:-amd64}" in \
        amd64) rust_host=x86_64-unknown-linux-gnu; rustup_sha="${RUSTUP_LINUX_AMD64_SHA256}" ;; \
        arm64) rust_host=aarch64-unknown-linux-gnu; rustup_sha="${RUSTUP_LINUX_ARM64_SHA256}" ;; \
        *) echo "Unsupported Rust TARGETARCH=${TARGETARCH:-unknown}" >&2; exit 1 ;; \
    esac; \
    curl --proto '=https' --tlsv1.2 -fsSLo /tmp/rustup-init \
        "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/${rust_host}/rustup-init"; \
    echo "${rustup_sha}  /tmp/rustup-init" | sha256sum -c -; \
    chmod +x /tmp/rustup-init; \
    /tmp/rustup-init -y --profile minimal --default-toolchain "${RUST_VERSION}" \
        --default-host "${rust_host}" --no-modify-path; \
    rm /tmp/rustup-init; \
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
# The digest is an official multi-platform uv image; the lockfile supplies hashes
# for all Python runtime and build dependencies.
COPY --from=ghcr.io/astral-sh/uv@sha256:2381d6aa60c326b71fd40023f921a0a3b8f91b14d5db6b90402e65a635053709 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock README.md /opt/sylliptor/
COPY src /opt/sylliptor/src
ENV UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PROJECT_ENVIRONMENT=/opt/sylliptor/.venv \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/sylliptor/.venv/bin:/home/sylliptor/.cargo/bin:/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN uv sync --locked --no-dev --no-editable --extra server; \
    uv cache clean; \
    sylliptor --version

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD git --version && rg --version && sylliptor --version
CMD ["sh", "-lc", "echo sylliptor-sandbox ready"]
WORKDIR /workspace
USER sylliptor

FROM ${VARIANT} AS final
USER sylliptor
