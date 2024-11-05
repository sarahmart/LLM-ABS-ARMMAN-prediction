#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <num_arms> <t1> <t2>"
    exit 1
fi

# Assign command-line arguments to variables
NUM_ARMS=$1
T1=$2
T2=$3

# Run the Python script in parallel with the specified configurations and arguments
python3 models/main.py --num_arms "$NUM_ARMS" --config_path openai_config.json --t1 "$T1" --t2 "$T2" &
python3 models/main.py --num_arms "$NUM_ARMS" --config_path anthropic_config.json --t1 "$T1" --t2 "$T2" &

# Wait for all background processes to finish
wait

echo "Both processes have completed."

