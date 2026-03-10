#!/bin/bash

# Export Google Application Credentials
export GOOGLE_APPLICATION_CREDENTIALS="gdelt/ism-gdelt-key.json"

# Move to the media-bias project directory
cd /home/cloud/project/media-bias || exit

# Run the pipeline script with nohup to run in the background
nohup python3 gdelt/gdelt_pipeline.py > gdelt/run_pipeline.log 2>&1 &

echo "GDELT pipeline started in the background. Check gdelt/run_pipeline.log for output."
