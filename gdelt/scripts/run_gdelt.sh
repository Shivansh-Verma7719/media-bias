#!/bin/bash

# Export Google Application Credentials
export GOOGLE_APPLICATION_CREDENTIALS="gdelt/ism-gdelt-key.json"

# Move to the media-bias project directory
cd /home/cloud/project/media-bias || exit

# Run the pipeline script
python3 gdelt/gdelt_pipeline.py
