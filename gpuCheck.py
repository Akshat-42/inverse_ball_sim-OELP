import torch

if torch.cuda.is_available():
    print("Success! PyTorch can see your GPU.")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("Failure! PyTorch cannot see your GPU.")
    print("Please follow the checklist to install drivers, the CUDA toolkit, and the correct PyTorch version.")