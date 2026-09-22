FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY scripts ./scripts
COPY src ./src

RUN pip install --no-cache-dir .

ENTRYPOINT ["hydrowatch-baseline"]
CMD ["--help"]
