"""One minute check that the Colab CLI can start a GPU machine and run code. No data, no keys."""
import platform, subprocess, time
t0 = time.time()
print("python", platform.python_version())
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
try:
    import torch
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        x = torch.randn(4096, 4096, device="cuda")
        torch.cuda.synchronize(); s = time.time()
        for _ in range(20):
            x = x @ x.T / 4096
        torch.cuda.synchronize()
        print("20 matrix multiplications took %.2f s" % (time.time() - s))
except ImportError:
    print("torch is not installed on this machine")
print("script seconds: %.1f" % (time.time() - t0))
