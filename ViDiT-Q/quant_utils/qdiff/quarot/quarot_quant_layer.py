import torch
import torch.nn as nn
import torch.nn.functional as F
from qdiff.base.quant_layer import QuantizedLinear
from qdiff.quarot.quarot_utils import random_hadamard_matrix, matmul_hadU_cuda

class QuarotQuantizedLinear(QuantizedLinear):
    """
    the base quantized linear layer,
    adpot the static weight quantization,
    and the dynamic activation quantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        device: None,
        quant_config: dict,
        fp_module: torch.nn.Linear,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, quant_config, fp_module)

        self.rotation_matrix = None   # init so could be load in quant_params

    def get_rotation_matrix(self):
        self.rotation_matrix = random_hadamard_matrix(self.in_features, "cuda")

    def update_quantized_weight_rotated(self):
        self.w_quantizer.init_done = False   # unset the init done to overwrite quant_params
        # modify:
        self.fp_module.weight.data = self.fp_module.weight.data.to("cuda")
        # self.rotation_matrix = self.rotation_matrix.to(device)
        # self.rotation_matrix = self.rotation_matrix.to("cpu")
        # rotated_weight = torch.matmul(self.fp_module.weight.data.double(), self.rotation_matrix).float()
        # self.weight.data = self.w_quantizer(rotated_weight)
        # print(f"[DEBUG] self.fp_module.weight.device: {self.fp_module.weight.data.device}")
        # print(f"[DEBUG] double() 后weight.device: {self.fp_module.weight.data.double().device}")
        # print(f"[DEBUG] self.rotation_matrix.device: {self.rotation_matrix.device}")
        self.weight.data = self.w_quantizer(torch.matmul(self.fp_module.weight.data.double(), self.rotation_matrix).float())
        
        # self.fp_module.weight.data = self.fp_module.weight.data.to("cpu")

        self.w_quantizer.init_done = True

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        input shape: [B,N_token,C]
        """
        if not self.quant_mode:  # use the FP
            return self.fp_module(x, *args, **kwargs)
        else:
            # reshape X into [G, -1] 
            B, N_token, C = x.shape
            # x = x.to("cpu")
            # self.rotation_matrix = self.rotation_matrix.to("cuda")
            # x_cpu = x.double().to("cpu")  # 转为 double 在 CPU 上 matmul
            # x = torch.matmul(x_cpu, self.rotation_matrix).to(dtype=x.dtype).to(x.device)  # 回到原来的设备和精度
            x = torch.matmul(x.double(), self.rotation_matrix).to(dtype=x.dtype)
            x = x.reshape([B*N_token,-1])

            # quantize activation
            x = self.a_quantizer(x)
            x = x.reshape([B, N_token, C])

            y = F.linear(x, self.weight.to(x.dtype), self.bias, *args, **kwargs)

            return y



