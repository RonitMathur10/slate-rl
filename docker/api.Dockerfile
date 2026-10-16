# FastAPI service. CPU-only PyTorch keeps the image small (no CUDA wheels).
# Trained artifacts and processed data are mounted at runtime (see docker-compose.yml).
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app

COPY pyproject.toml uv.lock README.md ./
RUN uv export --frozen --no-dev --group serve --no-hashes --no-emit-project -o requirements.txt \
 && TORCH_VERSION=$(sed -n 's/^torch==\([^ ;]*\).*/\1/p' requirements.txt) \
 && grep -vE '^(nvidia-|triton|torch==|cuda-)' requirements.txt > requirements-cpu.txt \
 && uv pip install --system --no-cache "torch==${TORCH_VERSION}" \
      --index-url https://download.pytorch.org/whl/cpu \
 && uv pip install --system --no-cache -r requirements-cpu.txt

COPY slate_rl ./slate_rl
COPY configs ./configs
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "slate_rl.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
