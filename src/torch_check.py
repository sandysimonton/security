import torch

# Instructions are specific to a windows/pip deployment
# Have to get the GPU enabled version of torch
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

print(torch.cuda.get_device_name())
print(torch.__version__)
print(torch.version.cuda)
x = torch.randn(1).cuda()
print(x)