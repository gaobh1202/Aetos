#!/bin/bash
# usage: apply.sh <OmniDrones checkout at BASE_COMMIT>
set -e
here="$(cd "$(dirname "$0")" && pwd)"
cd "$1"
git apply --check "$here/aetos.patch"
git apply "$here/aetos.patch"
cp -rn "$here/overlay/." .
echo "applied aetos.patch and overlay onto $(pwd)"
