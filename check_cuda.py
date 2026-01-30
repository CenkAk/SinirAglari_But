import torch

print("="*60)
print("PyTorch GPU KONTROLÜ")
print("="*60)

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
    
    device_count = torch.cuda.device_count()
    print(f"\nGPU Sayısı: {device_count}")
    
    for i in range(device_count):
        print(f"\n--- GPU {i} ---")
        print(f"  İsim: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Bellek: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi Processors: {props.multi_processor_count}")
    print("\n--- GPU Testi ---")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✓ GPU matrix çarpımı başarılı!")
        print(f"  Result shape: {z.shape}")
        del x, y, z
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"✗ GPU testi başarısız: {e}")
else:
    print("\n✗ CUDA kullanılamıyor!")
    print("  GPU eğitimi yapılamaz, CPU kullanılacak.")

print("\n" + "="*60)
