#!/usr/bin/env bash

cd "$(dirname "$0")" || exit
mkdir -p dist

bun run build.ts
