#!/usr/bin/env bash
# update_pypi.sh — build and upload BEoRN to PyPI
#
# Usage:
#   ./update_pypi.sh
#
# Before running:
#   1. Merge the release PR into main
#   2. Update the version in pyproject.toml
#   3. Move/create the git tag on main (see step 5 below)
#
# Requires: build, twine  (pip install build twine)
# PyPI credentials: use __token__ as username and your API token as password

set -euo pipefail

VERSION=$(python -c "import tomllib; f=open('pyproject.toml','rb'); d=tomllib.load(f); print(d['project']['version'])")
echo "Releasing version: $VERSION"

# 1. Make sure we are on main and up to date
git checkout main
git pull

# 2. Clean previous builds
rm -rf dist/ build/

# 3. Build source + wheel
pip install build
python -m build

# 4. Upload to PyPI (will prompt for credentials)
pip install twine
twine upload dist/*

# 5. Re-tag current main commit and push
git tag -f "v${VERSION}"
git push origin "v${VERSION}" --force

# 6. Create GitHub release
gh release create "v${VERSION}" dist/* \
    --title "v${VERSION}" \
    --notes "See docs/changelog.rst for details."

echo "Done — v${VERSION} is live on PyPI and GitHub."
