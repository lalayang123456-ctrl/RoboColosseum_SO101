#!/usr/bin/env bash
# Download all files of a public Hugging Face model repo into checkpoints/<repo-name>/,
# without needing git, git-lfs, or huggingface-cli (none are installed on this machine).
#
# Usage: scripts/download_hf_checkpoint.sh <org>/<repo>

set -euo pipefail

REPO_ID="${1:?Usage: $0 <org>/<repo>}"
REPO_NAME="${REPO_ID#*/}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT_DIR}/checkpoints/${REPO_NAME}"
BASE_URL="https://huggingface.co/${REPO_ID}/resolve/main"

mkdir -p "${DEST}"

FILES=$(wget -q -O - "https://huggingface.co/api/models/${REPO_ID}" \
  | python3 -c 'import json,sys; print("\n".join(f["rfilename"] for f in json.load(sys.stdin)["siblings"]))')

echo "Downloading ${REPO_ID} -> ${DEST}"
while IFS= read -r f; do
  echo "  ${f}"
  mkdir -p "${DEST}/$(dirname "${f}")"
  wget -q "${BASE_URL}/${f}" -O "${DEST}/${f}"
done <<< "${FILES}"

echo "Done: ${DEST}"
