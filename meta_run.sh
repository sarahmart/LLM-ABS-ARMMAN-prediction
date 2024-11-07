#!/bin/bash

for X in {0..9}
do
    python3 models/main.py --num_arms 100 --config_path meta_config.json --t1 $X --t2 $X
done

echo "All runs completed."