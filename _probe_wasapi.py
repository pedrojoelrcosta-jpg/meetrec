import sys
sys.path.insert(0, r"C:\Users\Utilizador\AppData\Local\Temp\pw_probe")
import pyaudiowpatch as pyaudio

p = pyaudio.PyAudio()
try:
    lb = p.get_default_wasapi_loopback()
    print("loopback:", lb["name"], "maxInputChannels=", lb["maxInputChannels"],
          "rate=", lb["defaultSampleRate"], "index=", lb["index"])
    mic = p.get_default_input_device_info()
    print("mic:", mic["name"], "maxInputChannels=", mic["maxInputChannels"],
          "rate=", mic["defaultSampleRate"])
    native = int(lb["maxInputChannels"])
    for ch in sorted({native, 1, 2, max(1, native - 1)}):
        try:
            s = p.open(format=pyaudio.paInt16, channels=ch,
                       rate=int(lb["defaultSampleRate"]), input=True,
                       input_device_index=lb["index"], frames_per_buffer=4096)
            s.close()
            print(f"  loopback channels={ch}: OK")
        except Exception as e:
            print(f"  loopback channels={ch}: FAIL {type(e).__name__}: {e}")
    try:
        s = p.open(format=pyaudio.paInt16, channels=native, rate=16000,
                   input=True, input_device_index=lb["index"],
                   frames_per_buffer=4096)
        s.close()
        print("  loopback rate=16000: OK")
    except Exception as e:
        print(f"  loopback rate=16000: FAIL {type(e).__name__}: {e}")
finally:
    p.terminate()
