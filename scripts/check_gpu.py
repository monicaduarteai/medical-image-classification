import torch

# Hardware Verification Utility for AI Research
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {device_name}")
    
    # Verify tensor allocation on the active CUDA device
    try:
        x = torch.randn(1, 3, 224, 224).cuda()
        print(f"Success: Tensor successfully moved to {device_name}!")
    except Exception as e:
        print(f"Error during tensor allocation: {e}")
else:
    print("CUDA is not available. Check your Miniconda environment and drivers.")