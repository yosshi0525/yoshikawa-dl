import torch
from torch import Tensor


a = Tensor(
    [
        [1, 2, 3],
        [4, 5, 6],
    ]
)


b = Tensor(
    [
        [7, 8, 9],
        [0, 1, 2],
    ]
)


torch.stack([a, b])


c = a.unsqueeze(dim=1)
d = b.unsqueeze(dim=1)


torch.cat([c, d])


c


for i in range(3):
    print(i)



