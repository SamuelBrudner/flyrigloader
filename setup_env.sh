#!/usr/bin/env bash
set -euo pipefail

DEV=0

for arg in "$@"; do
  case "$arg" in
    --dev)
      DEV=1
      shift
      ;;
  esac
done

conda env create -n flyrigloader --file environment.yml

if [[ $DEV -eq 1 ]]; then
  conda run -n flyrigloader pip install -e .[dev]
fi
