import torch, time

print("PyTorch:", torch.__version__)
print("CUDA dispo:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(8000, 8000, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    _ = x @ x.T
    torch.cuda.synchronize()
    print("Matmul OK en", round(time.time()-t0, 3), "s, VRAM:", torch.cuda.memory_allocated()//(1024*1024), "MB")
