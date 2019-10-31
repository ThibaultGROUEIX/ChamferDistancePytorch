from torch import nn
from torch.autograd import Function
import torch
import importlib
chamfer_found = importlib.find_loader("chamfer") is not None
if not chamfer_found:
    ## Cool trick from https://github.com/chrdiller
    from torch.utils.cpp_extension import load
    chamfer = load(name="chamfer",
          sources=["chamfer_cuda.cpp",
                   "chamfer.cu"])
    print("Loaded JIT CUDA chamfer distance")

else:
    import chamfer
    print("Loaded compiled CUDA chamfer distance")

# Chamfer's distance module @thibaultgroueix
# GPU tensors only
class chamferFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        device = xyz1.device

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.to(device)
        dist2 = dist2.to(device)
        idx1 = idx1.to(device)
        idx2 = idx2.to(device)
        torch.cuda.set_device(device)

        chamfer.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        device = graddist1.device

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.to(device)
        gradxyz2 = gradxyz2.to(device)
        chamfer.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2

class chamferDist(nn.Module):
    def __init__(self):
        super(chamferDist, self).__init__()

    def forward(self, input1, input2):
        return chamferFunction.apply(input1, input2)

