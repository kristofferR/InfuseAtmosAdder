#!/usr/bin/env bash
set -euo pipefail

APP_DIR=${APP_DIR:-/app}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FFMPEG_DIR=${FFMPEG_DIR:-${APP_DIR}/.ffmpeg}
VENV_PATH=${VENV_PATH:-${APP_DIR}/.venv}

mkdir -p "${FFMPEG_DIR}"

FFMPEG_BIN="${FFMPEG_DIR}/ffmpeg"
if [[ ! -x "${FFMPEG_BIN}" ]]; then
  echo "[bootstrap] downloading static FFmpeg build into ${FFMPEG_DIR}" >&2
  python "${SCRIPT_DIR}/download_ffmpeg.py" "${FFMPEG_DIR}"
fi
export PATH="${FFMPEG_DIR}:${PATH}"

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "[bootstrap] creating virtualenv at ${VENV_PATH}" >&2
  python -m venv "${VENV_PATH}"
  "${VENV_PATH}/bin/pip" install --upgrade pip
  "${VENV_PATH}/bin/pip" install --no-cache-dir fastapi uvicorn[standard] httpx
fi

exec "${VENV_PATH}/bin/uvicorn" app:app --host 0.0.0.0 --port "${PORT:-9999}" --workers 1
