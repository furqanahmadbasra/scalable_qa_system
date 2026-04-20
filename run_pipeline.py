import subprocess
import sys
import os

def run_script(script_path):
    print(f"\n{'='*50}")
    print(f"Running: {script_path}")
    print(f"{'='*50}\n")
    
    try:
        # Run the script and stream the output to the console
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True
        )
        process.wait()
        
        if process.returncode != 0:
            print(f"\n[ERROR] {script_path} failed with exit code {process.returncode}")
            sys.exit(process.returncode)
        else:
            print(f"\n[SUCCESS] Successfully finished: {script_path}")
            
    except Exception as e:
        print(f"\n[ERROR] Failed to run {script_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure we are in the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # List of scripts in the order they should be executed
    scripts_to_run = [
        os.path.join("src", "data_ingestion.py"),
        os.path.join("src", "indexing.py"),
        os.path.join("src", "lsh_indexing.py"),
        os.path.join("src", "evaluate_all.py")
    ]
    
    print(">>> Starting Scalable NUST QA Pipeline...\n")
    
    for script in scripts_to_run:
        if not os.path.exists(script):
            print(f"[ERROR] Could not find script at {script}")
            sys.exit(1)
        run_script(script)
        
    print(f"\n{'='*50}")
    print("[SUCCESS] All steps of the pipeline finished successfully!")
    print("View the results in experiments/results/evaluation_log.txt")
    print(f"{'='*50}\n")
