import os
import sys

def main():
    print("Launching Frontend...")
    import subprocess
    print("Launching Frontend...")
    subprocess.check_call([sys.executable, "-m", "streamlit", "run", "frontend/app.py"])

if __name__ == "__main__":
    main()
