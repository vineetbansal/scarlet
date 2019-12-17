import numpy as np
import torch


class MyF(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # ctx.save_for_backward(input)
        # return input.clamp(min=0)
        return input + 1

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # ctx.save_for_backward(input)
        # return input.clamp(min=0)
        grad_input = grad_output.clone()
        return grad_input

myf = MyF.apply


def myf2(x):
    r = np.random.random()
    print('random number = {}'.format(r))
    if r > 0.5:
        return x * 3
    else:
        return x * 2


if __name__ == '__main__':

    # Creating the graph
    x = torch.tensor(1.0, requires_grad = True)
    y = torch.tensor(3.0)
    z = x * y
    a = myf2(z)

    # Displaying
    for i, name in zip([x, y, z, a], "xyza"):
        print(f"{name}\ndata: {i.data}\nrequires_grad: {i.requires_grad}\n\
    grad: {i.grad}\ngrad_fn: {i.grad_fn}\nis_leaf: {i.is_leaf}\n")

    # x.grad is None unless we call 'backward' on a node
    a.backward()
    print(x.grad.data)
