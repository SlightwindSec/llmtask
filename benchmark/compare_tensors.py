import torch
import torch.nn.functional as F

def compare_tensors(tensor1, tensor2):
    # 检查两个tensor的尺寸是否一致
    if tensor1.size() != tensor2.size():
        raise ValueError("两个tensor的尺寸不匹配")

    # 1. 计算绝对平均误差 (Mean Absolute Error, MAE)
    mae = torch.mean(torch.abs(tensor1 - tensor2))

    # 2. 计算余弦相似度 (Cosine Similarity)
    # 展平成一维向量
    tensor1_flat = tensor1.view(-1)
    tensor2_flat = tensor2.view(-1)
    cosine_similarity = F.cosine_similarity(tensor1_flat.unsqueeze(0), tensor2_flat.unsqueeze(0)).item()

    # 3. 计算相对误差 (Relative Error)
    relative_error = torch.mean(torch.abs(tensor1 - tensor2) / (torch.abs(tensor1) + 1e-8))

    return {
        'MAE': mae.item(),
        'Cosine Similarity': cosine_similarity,
        'Relative Error': relative_error.item()
    }

if __name__ == '__main__':
    tensor_a = torch.randn(2, 3)
    tensor_b = torch.randn(2, 3)

    result = compare_tensors(tensor_a, tensor_b)
    print(result)
