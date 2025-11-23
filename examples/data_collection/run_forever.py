# run_forever.py
import subprocess
import time

PYTHON = r"C:/Users/qiyuanxu/AppData/Local/miniconda3/envs/mmf/python.exe"
BASE_DIR = r"C:\Users\qiyuanxu\Documents\GitHub\OptiBeam"
SCRIPT = r"c:/Users/qiyuanxu/Documents/GitHub/OptiBeam/examples/data_collection/data_collection_dmd.py"

DELAY = 2  # seconds between restarts
total_data = 250000
exit_code = 0

while int(exit_code) < total_data:
    # exactly what VSCode does, but in a loop
    exit_code = subprocess.call([PYTHON, SCRIPT], cwd=BASE_DIR)
    print(f"Script exited with code {exit_code}, restarting in {DELAY}s...")
    try:
        time.sleep(DELAY)
    except KeyboardInterrupt:
        print("Supervisor stopped by user.")
        break
