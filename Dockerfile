# Use the official uv image with Python 3.12 pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set the working directory in the container
WORKDIR /app

# Enable bytecode compilation for faster startup times
ENV UV_COMPILE_BYTECODE=1

# Use copy mode instead of hardlinks when using cache mounts
ENV UV_LINK_MODE=copy

# PyTorch build to install: "cpu" or "cu128" (GPU); override with --build-arg
ARG TORCH_VARIANT=cpu

# Install dependencies first (separate layer for better caching)
# --mount=type=cache: reuses downloaded packages between builds
# --mount=type=bind: temporarily mounts files without copying them
# --no-install-project: First installs only dependencies, then the project
# --no-dev: Excludes development dependencies from production image
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --extra ${TORCH_VARIANT}

# Copy the rest of the application code
ADD . /app

# Install the project itself (dependencies are already cached in the layer above)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ${TORCH_VARIANT}

# Run the application with uvicorn, binding to all interfaces on port 8000
# --no-sync: the environment is already complete, do not touch it at startup
CMD ["uv", "run", "--no-sync", "uvicorn", "ml_service.app:app", "--host", "0.0.0.0", "--port", "8000"]