#!/bin/sh
set -eu

CFG=/etc/garage.toml
CREDS=./bootstrap/creds.env
BUCKET=mlflow-artifacts
KEYNAME=mlflow-artifacts-key

GARAGE() { docker exec -i garage /garage -c "$CFG" "$@"; }

echo "Waiting for Garage to be ready..."
i=0
until GARAGE status >/dev/null 2>&1; do
  i=$((i+1))
  if [ "$i" -gt 120 ]; then
    echo "ERROR: Garage did not become ready in time."
    exit 1
  fi
  sleep 2
done

NODE_ID="$(GARAGE status | grep -Eo '([0-9a-f]{16,64})' | head -n 1 || true)"
if [ -z "$NODE_ID" ]; then
  echo "ERROR: Could not parse NODE_ID."
  exit 1
fi

# Assign role (idempotent)
GARAGE layout assign -z dc1 -c 10GB "$NODE_ID" >/dev/null 2>&1 || true

# Apply the version Garage *suggests* (robust parse)
NEW_VER="$(GARAGE layout show | grep -Eo 'apply --version [0-9]+' | tail -n 1 | grep -Eo '[0-9]+' || true)"
if [ -n "$NEW_VER" ]; then
  GARAGE layout apply --version "$NEW_VER" >/dev/null 2>&1 || true
fi

# Create bucket (idempotent)
GARAGE bucket create "$BUCKET" >/dev/null 2>&1 || true

# Keep creds if already real
NEED_NEW_KEY=1
if [ -f "$CREDS" ]; then
  # shellcheck disable=SC1090
  . "$CREDS" || true
  if [ "${AWS_ACCESS_KEY_ID:-}" != "PLACEHOLDER" ] && [ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    NEED_NEW_KEY=0
  fi
fi

if [ "$NEED_NEW_KEY" -eq 1 ]; then
  OUTTXT="$(GARAGE key create "$KEYNAME")"

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

# Allow the key on the bucket (retry a few times in case layout is still settling)
# shellcheck disable=SC1090
. "$CREDS"
j=0
until GARAGE bucket allow "$BUCKET" --key "$AWS_ACCESS_KEY_ID" --read --write --owner >/dev/null 2>&1; do
  j=$((j+1))
  if [ "$j" -gt 30 ]; then
    echo "ERROR: bucket allow did not succeed in time."
    exit 1
  fi
  sleep 1
done

echo "Garage bootstrap complete."