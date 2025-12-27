#!/usr/bin/env bash



cd "$(dirname "$0")" || exit
cd .. || exit

bun install --frozen-lockfile

./build/make-binary.sh

cp ./dist/noteview ~/.local/bin
echo "Copied noteview to $HOME/.local/bin"
