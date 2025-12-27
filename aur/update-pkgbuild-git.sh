#!/usr/bin/env bash

# Helper script to update PKGBUILD-git for a new release

if [ -z "$1" ]; then
    echo "Usage: ./update-pkgbuild-git.sh <version>"
    echo "Example: ./update-pkgbuild-git.sh 1.0.0"
    exit 1
fi

VERSION=$1
URL="https://github.com/FireSquid6/markdown-preview-server/archive/v${VERSION}.tar.gz"

echo "Downloading source tarball to calculate checksum..."
TEMP_FILE=$(mktemp)
curl -L -o "$TEMP_FILE" "$URL"

if [ $? -ne 0 ]; then
    echo "Error: Failed to download source from $URL"
    echo "Make sure the release tag exists on GitHub!"
    rm "$TEMP_FILE"
    exit 1
fi

CHECKSUM=$(sha256sum "$TEMP_FILE" | cut -d' ' -f1)
rm "$TEMP_FILE"

echo ""
echo "Version: $VERSION"
echo "SHA256: $CHECKSUM"
echo ""

# Update PKGBUILD-git
sed -i "s/^pkgver=.*/pkgver=${VERSION}/" PKGBUILD-git
sed -i "s/^pkgrel=.*/pkgrel=1/" PKGBUILD-git
sed -i "s/^sha256sums=.*/sha256sums=('${CHECKSUM}')/" PKGBUILD-git

echo "PKGBUILD-git updated!"
