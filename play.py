"""Launch the checkers UI server. Open http://localhost:8000 to play."""
import subprocess, sys, webbrowser, time

def main():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=__file__.replace("play.py", ""),
    )
    time.sleep(1.2)
    webbrowser.open("http://localhost:8000")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()

if __name__ == "__main__":
    main()
