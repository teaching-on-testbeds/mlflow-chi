#!/bin/sh
set -eu

CFG=/etc/garage.toml
CREDS=/bootstrap/creds.env
BUCKET=mlflow-artifacts
KEYNAME=mlflow-artifacts-key

echo "Waiting for Garage to be ready..."
i=0
until /garage -c "$CFG" status >/dev/null 2>&1; do
  i=$((i+1))
  if [ "$i" -gt 120 ]; then
    echo "ERROR: Garage did not become ready in time."
    exit 1
  fi
  sleep 2
done

# Grab node id
NODE_ID="$(/garage -c "$CFG" status | grep -Eo '([0-9a-f]{16,64})' | head -n 1 || true)"
if [ -z "$NODE_ID" ]; then
  echo "ERROR: Could not parse NODE_ID."
  exit 1
fi

# Single-node layout init (safe to re-run)
(/garage -c "$CFG" layout assign -z dc1 -c 10GB "$NODE_ID") || true

# Apply staged layout using the version Garage expects
NEW_VER="$(/garage -c "$CFG" layout show 2>/dev/null | grep -Eo 'version[[:space:]]+[0-9]+' | tail -n 1 | grep -Eo '[0-9]+' || true)"
if [ -n "$NEW_VER" ]; then
  (/garage -c "$CFG" layout apply --version "$NEW_VER") || true
fi

# Create bucket (safe to re-run)
(/garage -c "$CFG" bucket create "$BUCKET") || true

# If creds already look real, keep them (so restarts don't rotate)
NEED_NEW_KEY=1
if [ -f "$CREDS" ]; then
  . "$CREDS" || true
  if [ "${AWS_ACCESS_KEY_ID:-}" != "PLACEHOLDER" ] && [ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    NEED_NEW_KEY=0
  fi
fi

if [ "$NEED_NEW_KEY" -eq 1 ]; then
  OUTTXT="$(/garage -c "$CFG" key create "$KEYNAME")"

  KEY_ID="$(echo "$OUTTXT" | grep -Eo 'GK[0-9A-Za-z]+' | head -n 1 || true)"
  SECRET="$(echo "$OUTTXT" | grep -Eo '[0-9a-f]{64}' | head -n 1 || true)"
  if [ -z "$KEY_ID" ] || [ -z "$SECRET" ]; then
    echo "ERROR: Could not parse key id / secret."
    exit 1
  fi

  cat > "$CREDS" <<EOF
AWS_ACCESS_KEY_ID=$KEY_ID
AWS_SECRET_ACCESS_KEY=$SECRET
AWS_DEFAULT_REGION=us-east-1
AWS_EC2_METADATA_DISABLED=true
EOF
else
  echo "creds.env already populated; not rotating keys."
fi

# Allow key on bucket
. "$CREDS"
/garage -c "$CFG" bucket allow "$BUCKET" --key "$AWS_ACCESS_KEY_ID" --read --write --owner

echo "Garage bootstrap complete."
