#!/bin/bash

DIR="ldgms"
cd "$DIR" || exit 1

for file in *; do
  newname=$(echo "$file" | sed -E 's/ukbb\.EUR\.(1kg_chr[^.]+)\.path_distance=[0-9.]+\.l1_pen=[0-9.]+\.maf=[0-9.]+\.ALL\.edgelist/\1.UKBB.edgelist/')
  if [ "$newname" != "$file" ]; then
    mv "$file" "$newname"
  fi
done
