#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${1:-.}"
DEST_FILE="${DEST_DIR%/}/pt_224.npz"

GS_PATH="gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz"
HTTPS_URL="https://storage.googleapis.com/vertex-model-garden-paligemma-us/paligemma/pt_224.npz"

mkdir -p "$DEST_DIR"

if [ -f "$DEST_FILE" ]; then
  echo "Found existing file: $DEST_FILE"
  exit 0
fi

downloaded=0
if command -v gsutil >/dev/null 2>&1; then
  echo "[1/3] Trying gsutil (public object, no auth required)..."
  gsutil -m cp "$GS_PATH" "$DEST_FILE" && downloaded=1 || true
fi

if [ "$downloaded" -eq 0 ] && command -v curl >/dev/null 2>&1; then
  echo "[2/3] Trying curl over HTTPS..."
  curl -fL "$HTTPS_URL" -o "$DEST_FILE" && downloaded=1 || true
fi

if [ "$downloaded" -eq 0 ] && command -v wget >/dev/null 2>&1; then
  echo "[3/3] Trying wget over HTTPS..."
  wget -O "$DEST_FILE" "$HTTPS_URL" && downloaded=1 || true
fi

if [ "$downloaded" -eq 0 ]; then
  echo "ERROR: failed to download with gsutil/curl/wget."
  exit 2
fi

# Quick verification with NumPy (no internet needed)
python3 - <<'PY' "$DEST_FILE"
import sys, numpy as np, os
p = sys.argv[1]
try:
    f = np.load(p, allow_pickle=False)
    keys = list(f.keys())
    ok = any(k.startswith('params/') for k in keys)
    size = os.path.getsize(p)
    print(f"Downloaded {p} ({size/1024/1024:.1f} MiB)")
    print("Example keys:", keys[:5])
    if not ok:
        print("WARNING: file does not look like a PaliGemma params archive (no 'params/' keys).")
except Exception as e:
    print("ERROR: numpy could not read the file:", e)
    sys.exit(3)
PY

# Print SHA256 (optional)
( command -v sha256sum >/dev/null && sha256sum "$DEST_FILE" ) || ( command -v shasum >/dev/null && shasum -a 256 "$DEST_FILE" ) || true

echo "Done. Saved to $DEST_FILE"
