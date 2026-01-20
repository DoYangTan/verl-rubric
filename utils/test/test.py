import torch
import numpy as np
from collections import defaultdict
from typing import Optional

def compute_grpo_assign_1_1_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    改进 1.1：分段奖励填充版，包含每一步的详细打印
    """
    with torch.no_grad():
        bsz, seq_len = token_level_rewards.shape
        print(f"\n{'='*20} 步骤 1: 原始输入 {'='*20}")
        print(f"原始 Sparse Rewards (BS={bsz}, Seq={seq_len}):\n{token_level_rewards}")

        # --- 步骤 2: 实现阶段性奖励填充 ---
        # 逻辑：将 [0, 0.2, 0, 0.5] 修正为 [0.2, 0.2, 0.5, 0.5] [cite: 9, 10]
        filled_rewards = token_level_rewards.clone()
        for i in range(bsz):
            valid_len = int(response_mask[i].sum().item())
            current_val = 0.0
            # 从后往前扫描 [cite: 25]
            for j in range(valid_len - 1, -1, -1):
                if filled_rewards[i, j] != 0:
                    current_val = filled_rewards[i, j]
                filled_rewards[i, j] = current_val
        
        filled_rewards = filled_rewards * response_mask
        print(f"\n{'='*20} 步骤 2: 阶段填充 (改进 1.1 核心) {'='*20}")
        print(f"将奖励回传到该阶段所有 Token:\n{filled_rewards}")

        # --- 步骤 3: Token 级组内优势计算 ---
        advantages = torch.zeros_like(filled_rewards)
        unique_indices = np.unique(index)
        
        for idx in unique_indices:
            group_mask = (index == idx)
            group_rewards = filled_rewards[group_mask]
            group_resp_mask = response_mask[group_mask]
            
            # 统计每一列（每个位置）
            count_active = group_resp_mask.sum(dim=0)
            mean = group_rewards.sum(dim=0) / (count_active + epsilon)
            
            # 计算标准差
            mean_sq = (group_rewards**2).sum(dim=0) / (count_active + epsilon)
            std = torch.sqrt(torch.clamp(mean_sq - mean**2, min=0.0))
            
            print(f"\n{'='*15} 步骤 3: 组内统计 (Index Group: {idx}) {'='*15}")
            print(f"每一列的均值 (Mean per Token):\n{mean}")
            print(f"每一列的标准差 (Std per Token):\n{std}")

            if norm_adv_by_std_in_grpo:
                group_adv = (group_rewards - mean.unsqueeze(0)) / (std.unsqueeze(0) + epsilon)
            else:
                group_adv = group_rewards - mean.unsqueeze(0)
                
            advantages[group_mask] = group_adv * group_resp_mask

        print(f"\n{'='*20} 步骤 4: 最终计算出的 Advantages {'='*20}")
        print(f"Final Token-level Advantages:\n{advantages}")

    return advantages, advantages

def compute_grpo_assign_1_2_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    改进 1.2 实现：累加奖励 (Accumulated Rewards) + Token 级组内对比 [cite: 16, 17]
    """
    with torch.no_grad():
        # --- 步骤 1: 实现累加奖励计算 (改进 1.2 核心) ---
        # 逻辑：将 [0, 0.2, 0, 0.5] 修正为 [0, 0.2, 0.2, 0.7] [cite: 17, 18]
        # 注意：这里是前向累加，代表模型每达成一个 Rubric，后续 Token 的“身价”都会上涨 
        bsz, seq_len = token_level_rewards.shape
        accumulated_rewards = torch.cumsum(token_level_rewards, dim=-1)
        
        # 确保非 response 区域（Padding）为 0
        accumulated_rewards = accumulated_rewards * response_mask
        
        # --- 步骤 2: Token 级组内优势计算 ---
        # 针对每一列（相同索引的 token）进行对比 [cite: 10]
        advantages = torch.zeros_like(accumulated_rewards)
        unique_indices = np.unique(index)
        
        for idx in unique_indices:
            group_mask = (index == idx)
            group_rewards = accumulated_rewards[group_mask]  # (group_size, seq_len)
            group_resp_mask = response_mask[group_mask]
            
            # 计算每一列的有效均值和标准差
            count_active = group_resp_mask.sum(dim=0)
            mean = group_rewards.sum(dim=0) / (count_active + epsilon)
            
            # 计算标准差: sqrt(E[X^2] - (E[X])^2)
            mean_sq = (group_rewards**2).sum(dim=0) / (count_active + epsilon)
            std = torch.sqrt(torch.clamp(mean_sq - mean**2, min=0.0))
            
            # 归一化计算优势值
            if norm_adv_by_std_in_grpo:
                group_adv = (group_rewards - mean.unsqueeze(0)) / (std.unsqueeze(0) + epsilon)
            else:
                group_adv = group_rewards - mean.unsqueeze(0)
                
            advantages[group_mask] = group_adv * group_resp_mask

    return advantages, advantages

def test_assign_1_1():
    # 模拟 8 个样本，长度为 15
    group_size = 8
    seq_len = 15
    index = np.array([505] * group_size) # 8个样本属于同一组 
    
    token_rewards = torch.zeros(group_size, seq_len)
    mask = torch.zeros(group_size, seq_len)
    
    # 构造更加复杂的采样数据
    # 样本 0-2 (优秀组): 早期拿分高，逻辑简洁
    for i in range(3):
        token_rewards[i, 3] = 0.7  # 第一阶段拿 0.7 [cite: 7]
        token_rewards[i, 8] = 0.9  # 第二阶段拿 0.9
        mask[i, :10] = 1
        
    # 样本 3-5 (普通组): 过程拿分一般
    for i in range(3, 6):
        token_rewards[i, 5] = 0.4  # 拿分较晚且分低
        token_rewards[i, 11] = 0.6
        mask[i, :13] = 1
        
    # 样本 6-7 (冗余组): 拿分极慢且分低 [cite: 15]
    for i in range(6, 8):
        token_rewards[i, 14] = 0.3 # 只有最后给了一点分
        mask[i, :15] = 1

    # 设置打印选项，防止矩阵太长被折叠
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)
    
    # 执行
    compute_grpo_assign_1_1_advantage(token_rewards, mask, index)

def test_assign_1_2():
    print("========== Testing GRPO 1.2 (Accumulated Reward) ==========")
    index = np.array([303, 303]) 
    
    # 模拟数据 [cite: 17]
    # 样本 0: 在位置 2 给 0.2，位置 5 给 0.5
    # 样本 1: 在位置 4 给 0.3，位置 6 给 0.4
    token_rewards = torch.zeros(2, 8)
    token_rewards[0, 2] = 0.2; token_rewards[0, 5] = 0.5
    token_rewards[1, 4] = 0.3; token_rewards[1, 6] = 0.4
    
    mask = torch.ones(2, 8)
    mask[0, 7:] = 0; mask[1, 8:] = 0
    
    # 计算
    # 注意：此时内部会执行 torch.cumsum，将奖励变为累加态 
    adv, _ = compute_grpo_assign_1_2_advantage(token_rewards, mask, index)
    
    # 逻辑验证：
    # 样本 0 在 index 2-4 的奖励应为 0.2
    # 样本 0 在 index 5 之后的奖励应为 0.2 + 0.5 = 0.7 
    
    print("原始输入奖励:")
    print(token_rewards)
    print("\n1.2 处理后的优势值 (Advantages):")
    print(adv)
    print("\n验证点：观察优势值是否在新的奖励点触发后发生了阶梯式跳变。")

if __name__ == "__main__":
    test_assign_1_2()