import sys, threading, traceback
sys.path.insert(0, r"C:\Users\Utilizador\AppData\Local\Temp\wt_probe")

def work():
    try:
        from windows_toasts import (InteractableWindowsToaster, Toast,
                                    ToastActivatedEventArgs, ToastButton)
        t = InteractableWindowsToaster("meetrec")
        toast = Toast(["Title", "Body"])
        toast.AddAction(ToastButton("Open folder", "Open folder"))
        toast.on_activated = lambda args: None
        print("worker thread: toaster+toast built OK; AUMID =", t.notifierAUMID)
    except Exception:
        traceback.print_exc()

th = threading.Thread(target=work)
th.start()
th.join()
