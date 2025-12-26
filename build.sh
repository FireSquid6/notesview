#!/usr/bin/env bash


cd "$(dirname "$0")" || exit
mkdir -p output
bun build --compile ./src/index.ts
