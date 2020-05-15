#!/bin/bash

echo "Syncing $1"

rsync pat.chormai@mpg-server:~/attacut/artifacts/$1/stats.csv ./stats/$1.csv