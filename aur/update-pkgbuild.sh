#!/usr/bin/env bash

# Helper script to update PKGBUILD for a new release

if [ -z "$1" ]; then
    echo "Usage: ./update-pkgbuild.sh <version>"
    echo "Example: ./update-pkgbuild.sh 1.0.0"
    exit 1
fi

VERSION=$1
URL="https://github.com/FireSquid6/markdown-preview-server/releases/download/v${VERSION}/mdserve-linux-x64"

echo "Downloading binary to calculate checksum..."
TEMP_FILE=$(mktemp)
curl -L -o "$TEMP_FILE" "$URL"

if [ $? -ne 0 ]; then
    echo "Error: Failed to download binary from $URL"
    echo "Make sure the release exists on GitHub!"
    rm "$TEMP_FILE"
    exit 1
fi

CHECKSUM=$(sha256sum "$TEMP_FILE" | cut -d' ' -f1)
rm "$TEMP_FILE"

echo ""
echo "Version: $VERSION"
echo "SHA256: $CHECKSUM"
echo ""

# Update PKGBUILD
sed -i "s/^pkgver=.*/pkgver=${VERSION}/" PKGBUILD
sed -i "s/^pkgrel=.*/pkgrel=1/" PKGBUILD
sed -i "s/^sha256sums_x86_64=.*/sha256sums_x86_64=('${CHECKSUM}')/" PKGBUILD

