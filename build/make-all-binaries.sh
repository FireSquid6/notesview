#!/usr/bin/env bash

cd "$(dirname "$0")" || exit
cd .. || exit

# Run the bundle script first
echo "Bundling application..."
bun run build/bundle.ts

# Create output directory for binaries
mkdir -p dist/binaries

echo ""
echo "Building binaries for all platforms..."

# Linux x64
echo "Building for Linux x64..."
bun build ./dist/index.js --compile --target=bun-linux-x64 --outfile ./dist/binaries/notesview-linux-x64

# Linux ARM64
echo "Building for Linux ARM64..."
bun build ./dist/index.js --compile --target=bun-linux-arm64 --outfile ./dist/binaries/notesview-linux-arm64

# macOS x64 (Intel)
echo "Building for macOS x64..."
bun build ./dist/index.js --compile --target=bun-darwin-x64 --outfile ./dist/binaries/notesview-macos-x64

# macOS ARM64 (Apple Silicon)
echo "Building for macOS ARM64..."
bun build ./dist/index.js --compile --target=bun-darwin-arm64 --outfile ./dist/binaries/notesview-macos-arm64

# Windows x64
echo "Building for Windows x64..."
bun build ./dist/index.js --compile --target=bun-windows-x64 --outfile ./dist/binaries/notesview-windows-x64.exe

echo ""
echo "All binaries created in: dist/binaries/"
echo "  - notesview-linux-x64"
echo "  - notesview-linux-arm64"
echo "  - notesview-macos-x64"
echo "  - notesview-macos-arm64"
echo "  - notesview-windows-x64.exe"
