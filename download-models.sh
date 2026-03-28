#!/usr/bin/env bash
# Download model and tokenizer files for llm-lab.
# These are from Karpathy's llama2.c project (tinystories 15M model).
set -euo pipefail

mkdir -p data

echo "Downloading tokenizer.bin..."
curl -L -o data/tokenizer.bin \
  "https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin"

echo "Downloading stories15M.bin (58 MB)..."
curl -L -o data/stories15M.bin \
  "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin"

echo "Done. Files in data/:"
ls -lh data/
