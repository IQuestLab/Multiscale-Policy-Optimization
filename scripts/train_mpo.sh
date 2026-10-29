#!/usr/bin/env bash
set -euo pipefail
export MPO_STAGE=train
exec bash "$(dirname "${BASH_SOURCE[0]}")/_launch_mpo.sh" "$@"
