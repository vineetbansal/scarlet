import torch


def mul(a, b):
    re1, im1 = torch.unbind(a, -1)
    re2, im2 = torch.unbind(b, -1)
    re = re1 * re2 - im1 * im2
    im = re1 * im2 + re2 * im1
    retval = torch.stack((re, im), -1)

    return retval


def truediv(a, b):
    re1, im1 = torch.unbind(a, -1)
    re2, im2 = torch.unbind(b, -1)
    denominator = re2 ** 2 + im2 ** 2
    denominator = torch.stack((denominator, denominator), -1)
    t1 = re1 * re2 + im1 * im2
    t2 = im1 * re2 - re1 * im2
    numerator = torch.stack((t1, t2), -1)
    retval = numerator / denominator

    return retval
