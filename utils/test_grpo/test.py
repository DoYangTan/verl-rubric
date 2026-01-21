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

def compute_grpo_assign_1_2_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    改进 1.2 实现：累加奖励 (Accumulated Rewards) + Token 级组内对比
    """
    with torch.no_grad():
        bsz, seq_len = token_level_rewards.shape
        print(f"\n{'='*30} GRPO 内部计算开始 {'='*30}")
        print(f"-> 步骤 1: 原始 Sparse Rewards (BS={bsz}, Seq={seq_len})")
        print(token_level_rewards)

        # --- 步骤 2: 实现累加奖励计算 ---
        # 使用 cumsum 实现前向累加，代表模型每达成一个 Rubric，后续 Token 的奖励基数都会上涨 
        accumulated_rewards = torch.cumsum(token_level_rewards, dim=-1)
        accumulated_rewards = accumulated_rewards * response_mask
        
        print(f"\n-> 步骤 2: 累加后的阶梯奖励 (Accumulated Rewards)")
        print(accumulated_rewards)

        # --- 步骤 3: Token 级组内优势计算 ---
        advantages = torch.zeros_like(accumulated_rewards)
        unique_indices = np.unique(index)
        
        for idx in unique_indices:
            group_mask = (index == idx)
            group_rewards = accumulated_rewards[group_mask]
            group_resp_mask = response_mask[group_mask]
            num_in_group = group_rewards.shape[0]
            
            # 计算每一列（每个位置）的有效均值和标准差 
            count_active = group_resp_mask.sum(dim=0)
            mean = group_rewards.sum(dim=0) / (count_active + epsilon)
            
            # 计算标准差
            mean_sq = (group_rewards**2).sum(dim=0) / (count_active + epsilon)
            std = torch.sqrt(torch.clamp(mean_sq - mean**2, min=0.0))
            
            print(f"\n{'-'*15} 组内统计 | Group ID: {idx} | 样本数: {num_in_group} {'-'*15}")
            print(f"  Token 级均值 (Mean):\n  {mean}")
            print(f"  Token 级标准差 (Std):\n  {std}")

            if norm_adv_by_std_in_grpo:
                group_adv = (group_rewards - mean.unsqueeze(0)) / (std.unsqueeze(0) + epsilon)
            else:
                group_adv = group_rewards - mean.unsqueeze(0)
            
            # 只保留 mask 为 1 的部分的优势值
            group_adv = group_adv * group_resp_mask
            advantages[group_mask] = group_adv
            
            print(f"  组内归一化后的 Advantage (部分):\n{group_adv}")

        print(f"\n{'='*30} 步骤 4: 最终 Advantage 矩阵 {'='*30}")
        print(advantages)

    return advantages, advantages

def test_assign_1_2():
    print("========== Testing GRPO 1.2 with 8 Samples ==========")
    
    # 修改为 8 条数据，分为两个组 303 和 304
    index = np.array([303, 303, 303, 303, 304, 304, 304, 304]) 
    
    # 模拟 8x10 的数据 (8个样本，序列长度10)
    seq_len = 10
    token_rewards = torch.zeros(8, seq_len)
    
    # 为不同样本设置一些随机的稀疏奖励
    # Group 303
    token_rewards[0, 2] = 0.5; token_rewards[0, 7] = 0.5  # 表现好
    token_rewards[1, 3] = 0.2                             # 表现一般
    token_rewards[2, 2] = 0.5                             # 表现中等
    token_rewards[3, 8] = 0.1                             # 表现差
    
    # Group 304
    token_rewards[4, 1] = 0.8                             # 早期拿高分
    token_rewards[5, 5] = 0.4; token_rewards[5, 9] = 0.4  # 后期拿分
    token_rewards[6, 4] = 0.2
    token_rewards[7, 2] = 0.3
    
    # 模拟不同的 Mask（有些序列较短）
    mask = torch.ones(8, seq_len)
    mask[0, 9:] = 0
    mask[3, 7:] = 0
    mask[6, 5:] = 0

    # 执行计算
    adv, _ = compute_grpo_assign_1_2_advantage(token_rewards, mask, index)
    
    print("\n" + "#"*50)
    print("测试结论验证：")
    print("1. 检查 '累加奖励' 是否让奖励在时间轴上向后传递。")
    print("2. 检查 Group 303 和 304 的优势值是否是独立计算的（组内对比）。")
    print("3. 观察在 Mask=0 的位置，Advantage 是否成功被置为 0。")

import torch
import numpy as np
import pandas as pd

# 设置 pandas 显示选项以方便观察矩阵
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:,.3f}'.format)

def compute_grpo_outcome_advantage_v1_3_final_reset(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    decay_factor: float = 0.90
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    改进 1.3 核心逻辑（严格修正版）：
    1. 遇到新奖励：processed_reward = 历史所有原始奖励之和（彻底抛弃之前的衰减残值）。
    2. 无奖励空窗期：当前值按衰减系数下降。
    """
    with torch.no_grad():
        bsz, seq_len = token_level_rewards.shape
        processed_rewards = torch.zeros_like(token_level_rewards)
        
        print(f"\n{'='*20} 步骤 1: 详细转换过程追踪 (Target: 0.6, 1.1) {'='*20}")
        
        for i in range(bsz):
            global_reward_sum = 0.0      # 纯粹的原始奖励累加器
            current_val = 0.0            # 该时刻呈现给模型的动态奖励值
            
            print(f"\n[处理样本 {i+1}]")
            for j in range(seq_len):
                if response_mask[i, j] == 0:
                    continue
                
                raw_new = token_level_rewards[i, j].item()
                
                if raw_new > 0:
                    # 关键逻辑：先更新全局总和
                    global_reward_sum += raw_new
                    # 核心重置：当前点直接等于全局总和，抛弃残值
                    old_residue = current_val 
                    current_val = global_reward_sum 
                    print(f"  Token {j:2d}: 【★奖励点】 原始加分:{raw_new:.2f} | 抛弃残值:{old_residue:.3f} -> 重置为全局总和:{current_val:.3f}")
                else:
                    # 空窗期衰减
                    current_val *= decay_factor
                    print(f"  Token {j:2d}: [空窗期] 衰减中 -> {current_val:.3f}")
                
                processed_rewards[i, j] = current_val

        # --- 步骤 2: 组内 Advantage 计算 ---
        advantages = torch.zeros_like(processed_rewards)
        unique_indices = np.unique(index)
        
        for idx in unique_indices:
            group_mask = (index == idx)
            group_rewards = processed_rewards[group_mask]
            group_resp_mask = response_mask[group_mask]
            
            # 列维度均值与标准差计算
            count_active = group_resp_mask.sum(dim=0).clamp(min=epsilon)
            mean = group_rewards.sum(dim=0) / count_active
            mean_sq = (group_rewards**2).sum(dim=0) / count_active
            std = torch.sqrt(torch.clamp(mean_sq - mean**2, min=0.0))
            
            if norm_adv_by_std_in_grpo:
                group_adv = (group_rewards - mean) / (std + epsilon)
            else:
                group_adv = (group_rewards - mean)
            advantages[group_mask] = group_adv * group_resp_mask

    return processed_rewards, advantages

def run_8_samples_simulation():
    bsz, seq_len = 8, 15
    index = np.zeros(bsz, dtype=int)
    token_rewards = torch.zeros(bsz, seq_len)
    mask = torch.ones(bsz, seq_len)
    
    # --- 8 个样本策略模拟 ---
    # 1. 干练型 (0.3 + 0.3 + 0.5 = 1.1)
    token_rewards[0, 3], token_rewards[0, 7], token_rewards[0, 11] = 0.3, 0.3, 0.5
    # 2. 极其啰嗦 (同样的奖励，但间距极大)
    token_rewards[1, 1], token_rewards[1, 9], token_rewards[1, 14] = 0.3, 0.3, 0.5
    # 3. 抢跑型 (一上来就拿完 1.1)
    token_rewards[2, 1] = 1.1
    # 4. 逻辑密集型 (步步有奖，每步 0.22)
    for t in [2, 4, 6, 8, 10]: token_rewards[3, t] = 0.22
    # 5. 后期爆发型 (前程废话)
    token_rewards[4, 12], token_rewards[4, 13], token_rewards[4, 14] = 0.3, 0.3, 0.5
    # 6. 半途而废 (只拿第一步)
    token_rewards[5, 2] = 0.3
    # 7. 全错对照组
    token_rewards[6, :] = 0.0
    # 8. 完美紧凑型 (开头快速拿完)
    token_rewards[7, 1], token_rewards[7, 2], token_rewards[7, 3] = 0.3, 0.3, 0.5

    processed, adv = compute_grpo_outcome_advantage_v1_3_final_reset(token_rewards, mask, index)
    
    # 打印最终对比结果
    print(f"\n{'='*30} 8 样本平均优势 (Advantage) 排名 {'='*30}")
    labels = ["干练型", "极其啰嗦", "抢跑作弊", "逻辑密集", "后期爆发", "半途而废", "全错", "完美紧凑"]
    avg_adv = adv.mean(dim=1)
    for i in torch.argsort(avg_adv, descending=True):
        print(f"排名: {labels[i]:<8} | 样本 {i+1} | 平均 Advantage: {avg_adv[i].item():.4f}")

if __name__ == "__main__":
    run_8_samples_simulation()

