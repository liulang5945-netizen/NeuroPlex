import subprocess
import sys
import time
import os

def find_pids_by_port(port):
    try:
        out = subprocess.check_output(['netstat', '-ano'], stderr=subprocess.DEVNULL).decode(errors='ignore')
    except Exception:
        return set()
    pids = set()
    for line in out.splitlines():
        if f':{port} ' in line or f':{port}\t' in line:
            parts = line.split()
            if parts:
                pids.add(parts[-1])
    return pids

def kill_pids(pids):
    for pid in pids:
        try:
            subprocess.run(['taskkill', '/PID', str(pid), '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

def start_uvicorn():
    cmd = [sys.executable, '-m', 'uvicorn', 'api.app:app', '--host', '127.0.0.1', '--port', '8000']
    logfile = open('server.log', 'a', encoding='utf-8')
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    proc = subprocess.Popen(cmd, stdout=logfile, stderr=logfile, creationflags=creationflags)
    return proc.pid

if __name__ == '__main__':
    pids = find_pids_by_port(8000)
    if pids:
        kill_pids(pids)
        time.sleep(1)
    pid = start_uvicorn()
    print(f"uvicorn started with PID {pid}")