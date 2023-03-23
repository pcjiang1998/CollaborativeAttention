import torch
import torch.nn as nn
from einops import rearrange, reduce


class ConAM(torch.nn.Module):
    def __init__(self, num_channels: int, patch_size: int, inner_features=None) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.check_flag = False
        self.num_channels = num_channels
        self.inner_features = (
            inner_features if inner_features is not None else self.num_channels
        )

        self.linear1 = nn.Linear(self.num_channels, self.inner_features)
        self.linear2 = nn.Linear(self.inner_features, self.num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def check(self, input: torch.Tensor) -> bool:
        input_c, input_h, input_w = input.size(1), input.size(2), input.size(3)
        assert input_c == self.num_channels
        assert (input_h // self.patch_size) == (input_h / self.patch_size)
        assert (input_w // self.patch_size) == (input_w / self.patch_size)
        self.check_flag = True
        return True

    def attention(self, input: torch.Tensor) -> torch.Tensor:
        if not self.check_flag and not self.check(input):
            raise RuntimeError("Input size can't be divided by patch size")
        global_feature = reduce(
            input, "b c h w -> b c", reduction="mean"
        )  # Global Average Pooling
        global_feature = torch.unsqueeze(global_feature, dim=1)  # Add patch dimension

        local_feature = rearrange(
            input,
            "b c (h h1) (w w1) -> b c h w h1 w1",
            h1=self.patch_size,
            w1=self.patch_size,
        )  # Divide by patch and save in 'input_', get (b,c,h/ps,w/ps,ps,ps)
        input_ = local_feature
        local_feature = reduce(
            local_feature, "b c h w h1 w1 -> b c h w", reduction="mean"
        )  # Pooling with 'size=self.patch_size and stride=self.patch_size' from input
        h = local_feature.shape[2]  # Remember the current height for later recovery
        local_feature = rearrange(
            local_feature, "b c h w -> b (h w) c"
        )  # (b,p,c) Merge patch dimension
        mix_local_global = torch.cat(
            [local_feature, global_feature], 1
        )  # (b,p+1,c) Add one patch from 'global_feature'

        # MLP ()
        mix_local_global = self.linear1(mix_local_global)
        mix_local_global = self.relu(mix_local_global)
        mix_local_global = self.linear2(mix_local_global)
        mix_local_global = self.relu(mix_local_global)

        local_feature, global_feature = torch.split(
            mix_local_global, [local_feature.shape[1], 1], dim=1
        )  # (b,p,c), (b,1,c), global_feature.shape[1] is always 1
        global_feature = rearrange(
            global_feature, "b p c -> b c p"
        )  # (b,c,1) Transpose

        attention = torch.matmul(
            local_feature, global_feature
        )  # (b,p,c)x(b,c,1)=(b,p,1)
        attention = torch.squeeze(attention, dim=2)
        attention = self.softmax(
            attention
        )  # Make info in patch dimension competitive, get (b,p)
        attention = rearrange(attention, "b (h w) -> b 1 h w", h=h)  # c=1,Recovery
        attention = rearrange(attention, "b c h w -> b c h w 1 1")  # (b,1,h,w,1,1)
        input_ = input_ * attention  # (b,c,h/ps,w/ps,ps,ps)* (b,1,h,w,1,1)
        input_ = rearrange(input_, "b c h w h1 w1 -> b c (h h1) (w w1)")
        input = input + input_  # shortcut
        return input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x)
