#!/bin/bash

# Loop through values from 0 to 9
for X in {0..9}
do
    # Run the python command with --t1 and --t2 set to X
    python3 models/main.py --num_arms 100 --config_path openai_config.json --t1 $X --t2 $X
done

