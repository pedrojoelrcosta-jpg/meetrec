import subprocess, sys
from pathlib import Path

pythonw = Path(r"C:\Program Files\My Venv\Scripts\pythonw.exe")
cmd = f'"{pythonw}" -m meetrec start'
print("list2cmdline:", subprocess.list2cmdline(
    ["schtasks", "/Create", "/F", "/SC", "ONLOGON", "/TN", "meetrec_probe_tmp",
     "/TR", cmd]))
r = subprocess.run(["schtasks", "/Create", "/F", "/SC", "ONLOGON",
                    "/TN", "meetrec_probe_tmp", "/TR", cmd],
                   capture_output=True, text=True, errors="replace")
print("create rc:", r.returncode, r.stdout.strip(), r.stderr.strip())
if r.returncode == 0:
    q = subprocess.run(["schtasks", "/Query", "/TN", "meetrec_probe_tmp", "/XML"],
                       capture_output=True, text=True, errors="replace")
    print(q.stdout)
    d = subprocess.run(["schtasks", "/Delete", "/F", "/TN", "meetrec_probe_tmp"],
                       capture_output=True, text=True, errors="replace")
    print("delete rc:", d.returncode, d.stdout.strip(), d.stderr.strip())
