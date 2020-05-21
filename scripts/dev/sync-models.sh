#!/bin/bash

echo "Syncing $1"

mkdir -p ./artifact-server/$1 && rsync -a --recursive pat.chormai@mpg-server:~/attacut/artifacts/$1/ ./artifact-server/$1
