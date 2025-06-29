# -------- Build stage --------
FROM python:3.11.10-slim-bookworm AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && \
    apt-get -y --no-install-recommends install git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt \
 && python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("hotchpotch/static-embedding-japanese")
PY

# -------- Runtime stage --------
FROM python:3.11.10-slim-bookworm AS runtime
ENV PORT=8080
WORKDIR /app

# Copy installed site-packages and binaries from build stage
COPY --from=build /usr/local /usr/local
# Copy HuggingFace cache (downloaded model)
COPY --from=build /root/.cache/huggingface /root/.cache/huggingface

# Copy application source
COPY main.py log_config.yaml entrypoint.sh ./
RUN chmod +x entrypoint.sh

EXPOSE ${PORT}
CMD ["./entrypoint.sh"]
