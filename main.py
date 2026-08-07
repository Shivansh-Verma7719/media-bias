#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
from pathlib import Path

# Set up paths
ROOT_DIR = Path(__file__).resolve().parent

# Define the sequence of scripts to execute
PIPELINE = {
    "augmentation": [
        "augmentation/media_cloud_augmentation_pipeline.py"
    ],
    "inference": [
        "relevance_classifier_v2/10_full_inference_ensemble.py"
    ],
    "sentiment": [
        "mtsc/mtsc_pipeline.py"
    ],
    "finance": [
        "finance/vix_pipeline.py"
    ],
    "var_modeling": [
        "VAR/step1_data_inventory.py",
        "VAR/step2_bias_index.py",
        "VAR/step3_gap_analysis.py",
        "VAR/step4_returns_construction.py",
        "VAR/step5_stationarity.py",
        "VAR/step6_structural_breaks.py",
        "VAR/step7_var_specification.py",
        "VAR/step8_granger_causality.py",
        "VAR/step8_panel_granger_causality.py"
    ],
    "panel_regressions": [
        "panel_regressions/rq-1_revised.py",
        "panel_regressions/rq-2_revised.py"
    ]
}

def run_script(script_path: str):
    """Executes a Python script as a subprocess and streams output."""
    full_path = ROOT_DIR / script_path
    if not full_path.exists():
        print(f"[!] Error: Script {full_path} not found.")
        sys.exit(1)
        
    print(f"\n{'='*60}")
    print(f"🚀 Running: {script_path}")
    print(f"{'='*60}\n")
    
    # Run the script and stream its output to stdout
    result = subprocess.run(
        [sys.executable, str(full_path)],
        cwd=ROOT_DIR,
        check=False  # We'll check the return code manually to provide a better error message
    )
    
    if result.returncode != 0:
        print(f"\n[!] Error: Script {script_path} failed with exit code {result.returncode}.")
        print("Pipeline execution halted.")
        sys.exit(result.returncode)
    
    print(f"\n[✓] Successfully completed: {script_path}")

def main():
    parser = argparse.ArgumentParser(description="Media Bias Pipeline Orchestrator")
    parser.add_argument("--skip-augmentation", action="store_true", help="Skip the MediaCloud data augmentation phase (long-running).")
    parser.add_argument("--skip-inference", action="store_true", help="Skip the DeBERTa relevance classification inference phase.")
    parser.add_argument("--skip-sentiment", action="store_true", help="Skip the NewsMTSC sentiment classification phase.")
    parser.add_argument("--skip-finance", action="store_true", help="Skip pulling market data from Yahoo Finance.")
    parser.add_argument("--skip-var", action="store_true", help="Skip the VAR modeling steps (steps 1-8).")
    parser.add_argument("--skip-regressions", action="store_true", help="Skip the panel regressions (RQ1, RQ2).")
    
    args = parser.parse_args()

    print("Starting Media Bias Orchestrator Pipeline...\n")
    
    phases_to_run = []
    if not args.skip_augmentation: phases_to_run.append("augmentation")
    if not args.skip_inference: phases_to_run.append("inference")
    if not args.skip_sentiment: phases_to_run.append("sentiment")
    if not args.skip_finance: phases_to_run.append("finance")
    if not args.skip_var: phases_to_run.append("var_modeling")
    if not args.skip_regressions: phases_to_run.append("panel_regressions")
    
    if not phases_to_run:
        print("All phases were skipped. Nothing to execute.")
        sys.exit(0)
        
    print("Phases to be executed:")
    for i, phase in enumerate(phases_to_run, 1):
        print(f"  {i}. {phase}")
    print("\n")
    
    for phase in phases_to_run:
        scripts = PIPELINE[phase]
        for script in scripts:
            run_script(script)

    print("\n🎉 All requested pipeline phases have successfully completed!")

if __name__ == "__main__":
    main()
