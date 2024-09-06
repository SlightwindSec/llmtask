import torch
from hqq.core.quantize import Quantizer
from bitsandbytes.functional import quantize_nf4, dequantize_nf4, quantize_fp4, dequantize_fp4

from compare_tensors import compare_tensors

assert torch.cuda.is_available(), "error: CUDA is not available"


def test_hqq_4bit_fp16(x):
    x_q, meta = Quantizer.quantize(x, 4, group_size=64, optimize=False)
    x_d = Quantizer.dequantize(x_q, meta)
    r = compare_tensors(x, x_d)
    print(r)


def test_bnb_nf4_fp16(x):
    x_q, state = quantize_nf4(x, blocksize=64)
    x_d = dequantize_nf4(x_q, state)
    r = compare_tensors(x, x_d)
    print(r)

def test_bnb_fp4_fp16(x):
    x_q, state = quantize_fp4(x, blocksize=64)
    x_d = dequantize_fp4(x_q, state)
    r = compare_tensors(x, x_d)
    print(r)

if __name__ == '__main__':
    x = torch.randn((1024, 4096), dtype=torch.float16, device="cuda:0") * 0.1
    test_hqq_4bit_fp16(x.clone())
    test_bnb_nf4_fp16(x.clone())
    test_bnb_fp4_fp16(x.clone())

