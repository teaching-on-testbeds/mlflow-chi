#!/bin/sh
set -eu

CFG=/etc/garage.toml
CREDS=/bootstrap/creds.env
BUCKET=mlflow-artifacts

GARAGE() { docker exec -i garage /garage -c "$CFG" "$@"; }

echo "Waiting for Garage RPC..."
until GARAGE status >/dev/null 2>&1; do
  sleep 1
done

# Get node ID
NODE_ID="$(GARAGE status | grep -Eo '([0-9a-f]{16,64})' | head -n1)"
[ -n "$NODE_ID" ]

# Assign capacity (idempotent)
GARAGE layout assign -z dc1 -c 10GB "$NODE_ID" >/dev/null 2>&1 || true

# Apply layout if needed
VER="$(GARAGE layout show | grep -Eo 'apply --version [0-9]+' | tail -n1 | awk '{print $3}' || true)"
[ -z "$VER" ] || GARAGE layout apply --version "$VER" >/dev/null 2>&1 || true

# Create bucket (idempotent)
GARAGE bucket create "$BUCKET" >/dev/null 2>&1 || true

# Create key
OUT="$(GARAGE key create mlflow-key-$(date +%s))"

KEY_ID="$(echo "$OUT" | grep -Eo 'GK[0-9A-Za-z]+' | head -n1)"
SECRET="$(echo "$OUT" | grep -Eo '[0-9a-f]{64}' | head -n1)"

if [ -z "$KEY_ID" ] || [ -z "$SECRET" ]; then
  echo "ERROR: failed to create Garage key"
  exit 1
fi

# Write creds (overwrite whatever was there)
cat >"$CREDS" <<EOF
AWS_ACCESS_KEY_ID=$KEY_ID
AWS_SECRET_ACCESS_KEY=$SECRET
AWS_DEFAULT_REGION=us-east-1
AWS_EC2_METADATA_DISABLED=true
EOF

# Allow key on bucket (idempotent)
GARAGE bucket allow "$BUCKET" \
  --key "$KEY_ID" --read --write --owner >/dev/null 2>&1 || true

echo "Garage bootstrap complete."

