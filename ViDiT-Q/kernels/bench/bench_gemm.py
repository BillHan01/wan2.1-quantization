import torch
import viditq_extension.qgemm as qgemm
import icecream
M = 8192
N = 4096
K = 4096
input = torch.randint(-80, 80, (M, K), dtype=torch.int8).to("cuda")
weight = torch.randint(-80, 80, (N, K), dtype=torch.int8).to("cuda")

zp_weight = torch.randint(-10, 10, (N,), dtype=torch.int16).to("cuda")
# zp_weight = torch.zeros(N, dtype=torch.int16).to("cuda")

scale_input = 0.01 * torch.rand(M, dtype=torch.float16).to("cuda") + 0.005
scale_weight = 0.1 * torch.rand(N, dtype=torch.float16).to("cuda") + 0.1
# bias = torch.tensor([0.0 for _ in range(N)], dtype=torch.float16).to("cuda")
bias = torch.rand(N, dtype=torch.float16).to("cuda") * 200

# sum the token dim
input_sum = (scale_input.view(-1, 1).to(torch.float32) * input.to(torch.float32)).sum(dim=1).to(torch.float16)

input_fp16 = input.to(torch.float16)
input_fp32 = input.to(torch.float32)
weight_fp16 = weight.to(torch.float16)
weight_fp32 = weight.to(torch.float32)

output = qgemm.w8a8_of16_bias_weight_asym(input, weight, bias, scale_input, scale_weight, input_sum, zp_weight)
output_gt = (torch.nn.functional.linear(input_fp32, weight_fp32) * scale_input.view(-1, 1).to(torch.float32) * scale_weight.view(1, -1).to(torch.float32) 
             + input_sum.view(-1, 1).to(torch.float32) * zp_weight.to(torch.float32).view(1, -1) * scale_weight.view(1, -1).to(torch.float32) 
             + bias.to(torch.float32)).to(torch.float16)

icecream.ic(torch.max(torch.abs(output - output_gt)))

num_iter = 100
num_warmup_iter = 20

input_fp16 = torch.randn((M, K), dtype=torch.float16, device="cuda")
weight_fp16 = torch.randn((N, K), dtype=torch.float16, device="cuda")

for _ in range(num_warmup_iter):
    torch.nn.functional.linear(input_fp16, weight_fp16, bias=bias)
    torch.nn.functional.linear(input_fp16, weight_fp16, bias=None)
    qgemm.w8a8_of16_bias_weight_asym(input, weight, bias, scale_input, scale_weight, input_sum, zp_weight)
    qgemm.w8a8_of16_bias_weight_sym(input, weight, bias, scale_input, scale_weight)
    qgemm.w8a8_of16_nobias_weight_sym_qserve(input, weight, scale_input, scale_weight)

torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for i in range(num_iter):
    torch.nn.functional.linear(input_fp16, weight_fp16)

end.record()
torch.cuda.synchronize()
avg_time = start.elapsed_time(end) / num_iter
TFLOPS = 2 * M * N * K / avg_time / 1e9
print(f"FP16: {avg_time:.2f}ms, {int(TFLOPS)}T")

start.record()
for i in range(num_iter):
    torch.nn.functional.linear(input_fp16, weight_fp16, bias=bias)

end.record()
torch.cuda.synchronize()
avg_time = start.elapsed_time(end) / num_iter
TFLOPS = 2 * M * N * K / avg_time / 1e9
print(f"FP16 with bias: {avg_time:.2f}ms, {int(TFLOPS)}T")

start.record()
for i in range(num_iter):
    qgemm.w8a8_of16_bias_weight_sym(input, weight, bias, scale_input, scale_weight)
end.record()
torch.cuda.synchronize()
avg_time = start.elapsed_time(end) / num_iter
TFLOPS = 2 * M * N * K / avg_time / 1e9
print(f"W8A8 int8 OF16 with bias: {avg_time:.2f}ms, {int(TFLOPS)}T")

start.record()
for i in range(num_iter):
    qgemm.w8a8_of16_bias_weight_asym(input, weight, bias, scale_input, scale_weight, input_sum, zp_weight)
end.record()
torch.cuda.synchronize()
avg_time = start.elapsed_time(end) / num_iter
TFLOPS = 2 * M * N * K / avg_time / 1e9
print(f"W8A8 int8 OF16 with bias and weight zero point: {avg_time:.2f}ms, {int(TFLOPS)}T")

start.record()
for i in range(num_iter):
    qgemm.w8a8_of16_nobias_weight_sym_qserve(input, weight, scale_input, scale_weight)
end.record()
torch.cuda.synchronize()
avg_time = start.elapsed_time(end) / num_iter
TFLOPS = 2 * M * N * K / avg_time / 1e9
print(f"W8A8 int8 Qserve: {avg_time:.2f}ms, {int(TFLOPS)}T")
