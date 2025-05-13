import torch
import viditq_extension.fused as fused_kernels
from icecream import ic
import argparse

def test_quant_kernel(token_len, in_feat):
    x = torch.randn((token_len, in_feat), dtype=torch.float16, device="cuda")
    input_scale_gt = x.abs().max(dim=1).values.to(torch.float32) / 127.0
    quanted_x_gt = torch.round(x.to(torch.float32) / input_scale_gt.view(-1, 1))

    output_sum_gt = (quanted_x_gt.to(torch.float16).sum(dim=1) * input_scale_gt.view(-1)).to(torch.float16)

    x = (quanted_x_gt.to(torch.float16) * input_scale_gt.view(-1, 1)).to(torch.float16)

    input_scale = torch.zeros_like(input_scale_gt, dtype=torch.float16).to("cuda")
    output_sum = torch.zeros_like(input_scale_gt, dtype=torch.float16).to("cuda")

    quanted_x = fused_kernels.quant_sum(x, output_sum, input_scale)

    ic(torch.max(torch.abs(quanted_x - quanted_x_gt)))
    ic(torch.max(torch.abs(input_scale - input_scale_gt)))
    ic(torch.max(torch.abs(output_sum - output_sum_gt)))

    input_scale_gt = torch.nn.functional.gelu(x, approximate='tanh').to(torch.float32).abs().max(dim=1).values.to(torch.float32) / 127.0
    quanted_x_gt = torch.round(torch.nn.functional.gelu(x, approximate='tanh').to(torch.float32) / input_scale_gt.view(-1, 1))
    output_sum_gt = (quanted_x_gt.to(torch.float16).sum(dim=1) * input_scale_gt.view(-1)).to(torch.float16)

    quanted_x = fused_kernels.gelu_quant_sum(x, output_sum, input_scale)

    ic(torch.max(torch.abs(quanted_x - quanted_x_gt)))
    ic(torch.max(torch.abs(input_scale - input_scale_gt)))
    ic(torch.max(torch.abs(output_sum - output_sum_gt)))

parser = argparse.ArgumentParser()
parser.add_argument("--token_len", type=int, default=8192)
parser.add_argument("--in_feat", type=int, default=1152)

args = parser.parse_args()


num_iter = 40
num_warmup_iter = 20

token_len = args.token_len
in_feat = args.in_feat

print(f"token len: {token_len} in feat: {in_feat}")

test_quant_kernel(token_len, in_feat)

input_args = []

x = [torch.randn(in_feat, dtype=torch.float16, device="cuda") for _ in range(num_iter)]
y = [torch.randn(in_feat, dtype=torch.float16, device="cuda") for _ in range(num_iter)]

for _ in range(num_iter + 1):
    quanted_x = torch.randint(-128, 127, (token_len, in_feat), dtype=torch.int8).to("cuda")

    input_scale = torch.tensor([0.001 for _ in range(token_len)], dtype=torch.float16).to("cuda")
    sum_output = torch.zeros((token_len, ), dtype=torch.float16).to("cuda")

    x = quanted_x.to(torch.float16) * input_scale.view(-1, 1)

    quanted_x = torch.zeros_like(quanted_x, dtype=torch.int8).to("cuda")
    input_scale = torch.zeros_like(input_scale, dtype=torch.float16).to("cuda")

    input_args.append((x, sum_output, input_scale))

for _ in range(num_warmup_iter):
    fused_kernels.quant_sum(*input_args[-1])
    fused_kernels.quant_sum_static(*input_args[-1])
    fused_kernels.gelu_quant_sum(*input_args[-1])
    torch.nn.functional.gelu(input_args[-1][0], approximate='tanh')

torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

for i in range(num_iter):
    fused_kernels.quant_sum(*input_args[i])

end.record()
torch.cuda.synchronize()
print(
    f"quant_sum average inference time: {start.elapsed_time(end) / num_iter:.4f} ms")

start.record()
for i in range(num_iter):
    fused_kernels.quant_sum_static(*input_args[i])
end.record()
torch.cuda.synchronize()
print(
    f"quant_sum_static average inference time: {start.elapsed_time(end) / num_iter:.4f} ms")

start.record()
for i in range(num_iter):
    fused_kernels.gelu_quant_sum(*input_args[i])

end.record()
torch.cuda.synchronize()
print(
    f"gelu_quant_sum average inference time: {start.elapsed_time(end) / num_iter:.4f} ms")

start.record()
for i in range(num_iter):
    torch.nn.functional.gelu(input_args[i][0], approximate='tanh')

end.record()
torch.cuda.synchronize()
print(
    f"torch gelu average inference time: {start.elapsed_time(end) / num_iter:.4f} ms"
)
