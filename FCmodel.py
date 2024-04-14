import torch
import torch.nn as nn
import torch.nn.functional as F

class Surrogate_BP_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad

def PoissonGen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp).cuda()
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))

class SNN_FC_BNTT(nn.Module):
    def __init__(self, num_steps, leak_mem=0.95, img_size=28, num_cls=10):
        super(SNN_FC_BNTT, self).__init__()

        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem

        # Adjusted input size for sequential MNIST (784)
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_cls)

        # Incorporating BatchNorm for each layer
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)

        # Setting up the threshold for each layer
        self.threshold = 1.0
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.threshold = self.threshold

    def forward(self, inp):
        # Converting inp to a compatible shape
        inp = inp.view(-1, 784)
        mem1 = mem2 = torch.zeros(inp.size(0), 256).cuda()
        mem3 = torch.zeros(inp.size(0), self.num_cls).cuda()

        for step in range(self.num_steps):
            cur_inp = PoissonGen(inp)
            # FC1
            mem1 = self.leak_mem * mem1 + self.bn1(self.fc1(cur_inp))
            out1 = self.spike_fn(mem1 - self.fc1.threshold)
            mem1 = torch.where(out1 > 0, torch.zeros_like(mem1), mem1)  # Reset
            # FC2
            mem2 = self.leak_mem * mem2 + self.bn2(self.fc2(out1))
            out2 = self.spike_fn(mem2 - self.fc2.threshold)
            mem2 = torch.where(out2 > 0, torch.zeros_like(mem2), mem2)  # Reset
            # FC3
            mem3 += self.fc3(out2)

        out_voltage = mem3 / self.num_steps
        return out_voltage
