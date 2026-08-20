import locale, subprocess, sys
print("preferred encoding:", locale.getpreferredencoding(False))
r = subprocess.run(["tasklist", "/FI", "PID eq 999999", "/NH"],
                   capture_output=True)
print("raw bytes:", r.stdout[:200])
try:
    print("text=True decode:", r.stdout.decode(locale.getpreferredencoding(False)))
except Exception as e:
    print("DECODE FAILED:", e)
print("oem decode:", r.stdout.decode("oem", errors="replace").strip())
r2 = subprocess.run(["tasklist", "/FI", f"PID eq {123456}", "/NH"],
                    capture_output=True, text=True, errors="replace")
print("no-match stdout repr:", repr(r2.stdout[:120]))
print("'123456' in out ->", "123456" in r2.stdout)
