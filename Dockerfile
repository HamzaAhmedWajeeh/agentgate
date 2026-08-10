# syntax=docker/dockerfile:1.9
#
# Two stages. The builder resolves the locked dependency set into a self-contained virtual
# environment; the runtime copies that environment and nothing else, so uv, the lockfile, and
# the build toolchain never reach the shipped image.

ARG PYTHON_VERSION=3.13
ARG UV_VERSION=0.11.30

# ----------------------------------------------------------------------------- builder

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
ARG UV_VERSION

# uv comes from PyPI rather than its published image: one registry to reach instead of two,
# and this stage is discarded, so the extra layer costs nothing in the shipped image.
RUN pip install --no-cache-dir "uv==${UV_VERSION}"

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies resolve from the lockfile alone, in their own layer. Source changes below this
# point do not invalidate it, which is the difference between a 3 second and a 90 second build.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ ./src/

# --no-editable so the runtime stage needs no source tree, only the virtual environment.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ----------------------------------------------------------------------------- runtime

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# Numeric uid so the security posture holds even where the image is run with a read-only
# passwd database or scanned by a policy engine that does not resolve names.
ARG UID=10001
RUN groupadd --system --gid ${UID} agentgate \
    && useradd --system --uid ${UID} --gid ${UID} --no-create-home --shell /usr/sbin/nologin agentgate

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AGENTGATE_ENVIRONMENT=docker

WORKDIR /app

COPY --from=builder --chown=${UID}:${UID} /app/.venv /app/.venv

# The retrieval corpus. Read-only data, not code, and not installed with the package -- so
# without this line AGENTGATE_CORPUS_PATH resolves to a directory that does not exist inside
# the image and every research branch fails. The offline suite cannot catch that: it runs from
# a checkout where `corpus/` is simply there.
COPY --chown=${UID}:${UID} corpus/ /app/corpus/

# Writable state lives here and nowhere else, so the rest of the filesystem can be mounted
# read-only. Compose mounts a volume over it.
RUN install -d -o ${UID} -g ${UID} /app/data

USER ${UID}:${UID}

# Prints the resolved configuration and exits non-zero if the environment is unrunnable.
# Replaced by the API server once that surface exists; no HEALTHCHECK is declared until there
# is an endpoint for it to call.
CMD ["python", "-m", "agentgate"]
