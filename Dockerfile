# -- Stage 1: Get the uv binary --
FROM ghcr.io/astral-sh/uv:latest AS uv_bin

# -- Stage 2: Final Runtime (Correct tag for Mac) --
FROM rayproject/ray:2.40.0-py311-aarch64

USER root
WORKDIR /app

# Copy uv binary
COPY --from=uv_bin /uv /uv/bin/uv
ENV PATH="/uv/bin:$PATH"

COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

COPY . .

USER ray
EXPOSE 8000 8265

CMD bash -c "ray start --head --dashboard-host=0.0.0.0 --block & \
    sleep 3 && \
    python launch_serve.py"
