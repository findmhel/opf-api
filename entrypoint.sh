#!/usr/bin/env bash
set -euo pipefail

echo "Starting OpenFinance API entrypoint..."

# Default paths
SKLEARN_MODEL_PATH=${SKLEARN_MODEL_PATH:-/opt/render/project/src/app/model/sk_model.joblib}
PARQUET_FALLBACK_CSV=${PARQUET_FALLBACK_CSV:-/opt/render/project/src/data/features.csv}

# Create parent directories
mkdir -p "$(dirname "$SKLEARN_MODEL_PATH")"
mkdir -p "$(dirname "$PARQUET_FALLBACK_CSV")"

# Download sklearn model if not present
if [ -n "${SKLEARN_MODEL_URL:-}" ] && [ ! -f "$SKLEARN_MODEL_PATH" ]; then
  echo "Downloading model from SKLEARN_MODEL_URL..."
  if curl -fsSL "$SKLEARN_MODEL_URL" -o "$SKLEARN_MODEL_PATH"; then
    echo "Model downloaded successfully to $SKLEARN_MODEL_PATH"
  else
    echo "WARNING: Failed to download model from $SKLEARN_MODEL_URL"
  fi
fi

# Download features CSV if not present
if [ -n "${FEATURES_URL:-}" ] && [ ! -f "$PARQUET_FALLBACK_CSV" ]; then
  echo "Downloading features.csv from FEATURES_URL..."
  if curl -fsSL "$FEATURES_URL" -o "$PARQUET_FALLBACK_CSV"; then
    echo "Features CSV downloaded successfully to $PARQUET_FALLBACK_CSV"
  else
    echo "WARNING: Failed to download features.csv from $FEATURES_URL"
  fi
fi

# Start the application
echo "Starting uvicorn server..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
