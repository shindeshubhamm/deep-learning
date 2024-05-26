import torch
import torch.nn as nn


# modify the edge detector kernel in such a way that
# it calculates the derivatives in x and y direction
edge_detector_kernel = torch.zeros(2, 1, 2, 2)
edge_detector_kernel[0, 0, :, :] = torch.tensor([[-1, 0], [1, 0]])
edge_detector_kernel[1, 0, :, :] = torch.tensor([[-1, 1], [0, 0]])


class Conv2d(nn.Module):
    def __init__(self, kernel, padding=0, stride=1):
        super().__init__()
        self.kernel = nn.Parameter(kernel)
        self.padding = ZeroPad2d(padding)
        self.stride = stride

    def forward(self, x):
        x = self.padding(x)
        C, H, W = x.shape
        cout, cin, h, w = self.kernel.shape
        hout = (H - h) // self.stride + 1
        wout = (W - w) // self.stride + 1

        xout = torch.zeros(cout, hout, wout)

        for i in range(hout):
            for j in range(wout):
                p = (
                    x[
                        :,
                        i * self.stride : i * self.stride + h,
                        j * self.stride : j * self.stride + w,
                    ]
                    * self.kernel
                )
                xout[:, i, j] = p.sum(dim=[1, 2, 3])

        return xout


class ZeroPad2d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return nn.functional.pad(
            x, (self.padding, self.padding, self.padding, self.padding)
        )
