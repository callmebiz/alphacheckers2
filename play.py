"""Launch the checkers UI server. Open http://localhost:8000 to play."""
import os, subprocess, sys, webbrowser, time

def main():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
    )
    time.sleep(1.2)
    webbrowser.open("http://localhost:8000")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()

if __name__ == "__main__":
    main()
