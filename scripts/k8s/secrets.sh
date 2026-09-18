#!/usr/bin/env bash
# ANTHROPIC_API_KEY from .env -> Secret iran-sentiment/iran-sentiment-env
# (only teacher-check needs it). Values are parsed with python-dotenv so the
# pod sees exactly what the venv sees (quotes stripped).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
kubectl create namespace iran-sentiment --dry-run=client -o yaml | kubectl apply -f -
"$REPO/.venv/bin/python" - "$REPO/.env" <<'PY' | kubectl apply -f -
import json, sys
from dotenv import dotenv_values
vals = {k: v for k, v in dotenv_values(sys.argv[1]).items() if v is not None and k == "ANTHROPIC_API_KEY"}
print(json.dumps({"apiVersion": "v1", "kind": "Secret",
                  "metadata": {"name": "iran-sentiment-env", "namespace": "iran-sentiment"},
                  "type": "Opaque", "stringData": vals}))
PY
kubectl -n iran-sentiment get secret iran-sentiment-env -o go-template='{{range $k, $_ := .data}}  {{$k}}{{"\n"}}{{end}}'
