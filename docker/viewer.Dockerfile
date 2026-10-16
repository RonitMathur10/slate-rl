# Streamlit viewer. No PyTorch: it reads the committed reports/ and calls the API.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md ./
RUN uv export --frozen --no-dev --no-hashes --no-emit-project -o all.txt \
 && uv export --frozen --only-group viewer --no-hashes --no-emit-project -o requirements.txt \
 && grep -E '^(polars|pyarrow)==' all.txt >> requirements.txt \
 && uv pip install --system --no-cache -r requirements.txt

COPY viewer ./viewer
COPY reports ./reports
EXPOSE 8501
CMD ["streamlit", "run", "viewer/app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
