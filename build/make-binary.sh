#!/usr/bin/env bash

cd "$(dirname "$0")" || exit
cd .. || exit

bun run build/bundle.ts
bun build ./dist/index.js --compile --outfile ./dist/noteview

echo "Binary created at: dist/noteview"
