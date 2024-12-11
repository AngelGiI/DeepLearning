import torch
import torch.nn.functional as F
from torch import nn

class Conv2DFunc(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward
    passes which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input_batch, kernel, stride=1, padding=0):
        """
        In the forward pass we receive a Tensor containing the input
        and return a Tensor containing the output. ctx is a context
        object that can be used to stash information for backward
        computation. You can cache arbitrary objects for use in the
        backward pass using the ctx.save_for_backward method.
        """
        # store objects for the backward
        ctx.save_for_backward(input_batch)
        ctx.save_for_backward(kernel)

        # Working out the different dimensions
        b, c_in, h_in, w_in = input_batch.size()
        c_out, h_k, w_k = kernel.size()
        h_out = (h_in-h_k+2*padding) // stride + 1
        w_out = (w_in-w_k+2*padding) // stride + 1

        # Extracting patches (U)
        u = F.unfold(input_batch, (h_k, w_k), dilation=1, padding=padding, stride=stride)
        b, k, p = u.size()

        # Computing Y'
        uflat = torch.permute(u, (0, 2, 1))
        uflat = uflat.reshape(b * p, k)
        weights = torch.randn(c_out, k)
        yprime = torch.inner(uflat, weights)  # (b*p, k) x (k, c_out) --> (b*p, c_out)

        # Store final U and W for the backward
        ctx.save_for_backward(u)
        ctx.save_for_backward(uflat)
        ctx.save_for_backward(weights)

        # Computing Y
        y = yprime.reshape(b, h_out, w_out, c_out)
        output_batch = torch.permute(y, (0, 3, 1, 2))
        return output_batch

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve stored objects
        input, kernel, uflat, weights, u = ctx.saved_tensors

        # Working out the different dimensions
        b, c_in, h_in, w_in = input.size()
        c_out, h_k, w_k = kernel.size()
        b, c_out, h_out, w_out = grad_output.size()
        b, k, p = u.size()
        # Gradient of Y'
        reorder_grad_output = torch.permute(grad_output, (0, 2, 3, 1))
        yprime_grad = reorder_grad_output.reshape(b*h_out*w_out, c_out)  # (b*p, c_out)

        # Gradient of W
        kernel_grad = torch.inner(yprime_grad.T, uflat.T)  # (c_out, b*p) x (b*p, k) --> (c_out, k)

        # Gradient of U
        u_grad = torch.matmul(yprime_grad.T, weights.T) # (b*p, c_out) x (c_out, k) --> (b*p, k)

        # Gradient of X
        u_grad_deflat = uflat.reshape(b, p, k)
        u_grad_orig_shape = torch.permute(u_grad_deflat, (0, 2, 1))
        fold = torch.nn.Fold((h_in, w_in), (h_k, w_k), dilation=1, padding=0, stride=1)
        input_batch_grad = fold(u_grad_orig_shape)

        return input_batch_grad, kernel_grad, None, None

input_batch = torch.randn(16, 3, 32, 32)
kernel = torch.randn(4, 5, 5)
out = Conv2DFunc.apply(input_batch, kernel)
print(out.shape)  # For this example, should be (16,4,28,28)
