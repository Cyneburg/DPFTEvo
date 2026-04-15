import random
import math
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
import numpy as np

class ProteinSequenceData(Dataset):
    def __init__(self, sequences, tokenizer, device=None):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.device = device
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def collate(self, raw_batch):
        sequences = self.tokenizer(raw_batch, return_tensors='pt', padding=True, return_length=True)
        return sequences.to(self.device)

class MutantSequenceData(Dataset):
    def __init__(self, protein, tokenizer, mask=False, device=None, pref_pairs=None):
        if mask:
            self.sequences = {}
            for positions in set(protein['df']['positions']):
                mutant = list(protein['wild_type'])
                for position in positions: # get masked mutant sequence
                    mutant[position] = '<mask>'
                self.sequences[positions] = ''.join(mutant)
        else:
            self.sequences = [protein['wild_type']]
        
        for key, value in protein['df'].items():
            setattr(self, key, value.to_list())
        self.tokenizer = tokenizer
        self.device = device
        self.pref_pairs = pref_pairs
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        if self.pref_pairs is not None:
            return self.wt_aas[idx], self.mt_aas[idx], self.positions[idx], self.DMS_score[idx], self.DMS_score_bin[idx], self.pref_pairs
        else:
            return self.wt_aas[idx], self.mt_aas[idx], self.positions[idx], self.DMS_score[idx], self.DMS_score_bin[idx]
    
    def collate(self, raw_batch):
        if len(raw_batch[0]) == 5:  # 无pref_pairs
            wt_aas, mt_aas, positions, scores, labels = zip(*raw_batch)
            pref_pairs = None
        else:  # 有pref_pairs
            wt_aas, mt_aas, positions, scores, labels, pref_pairs = zip(*raw_batch)
            pref_pairs = torch.cat([p for p in pref_pairs if p is not None])  # 合并所有batch的
        
        if type(self.sequences) is dict: # identify duplicate positions, possibly multi-site
            unique_pos = {pos: i for i, pos in enumerate(set(positions))}
            inv_idx = torch.tensor([unique_pos[pos] for pos in positions], device=self.device)
            sequences = [self.sequences[pos] for pos in unique_pos.keys()]
        else:
            inv_idx = torch.zeros(len(positions), dtype=torch.long, device=self.device)
            sequences = self.sequences
        sequences = self.tokenizer(sequences, return_tensors='pt').to(self.device)
        
        positions = [torch.tensor(pos, device=self.device) + 1 for pos in positions]
        wt_aas = self.tokenizer(wt_aas, add_special_tokens=False)['input_ids']
        mt_aas = self.tokenizer(mt_aas, add_special_tokens=False)['input_ids']
        scores = torch.tensor(scores, device=self.device)
        labels = torch.tensor(labels, device=self.device)
        return dict(sequences=sequences,
                    inv_seq_idx=inv_idx,
                    wt_aas=wt_aas,
                    mt_aas=mt_aas,
                    positions=positions,
                    targets=scores,
                    labels=labels,
                    pref_pairs=pref_pairs)

class RankingSequenceData(Dataset):
    def __init__(self, protein, tokenizer, mask=True, list_size=2, max_size=10000,
                 constructor=MutantSequenceData, pref_pairs=None, device=None, active_learning=False):
        self.mutant_data = constructor(protein, tokenizer, mask, device)
        self.list_size = list_size
        self.max_size = max_size
        self.device = device
        self.active_learning = active_learning  # 标记是否处于主动学习阶段
        self.current_pref_pairs = pref_pairs  # 存储当前迭代的偏好对
        
        total = math.comb(len(self.mutant_data), list_size)
        if max_size > total: # iteration over all combinations
            self.comb_idx = list(combinations(range(len(self.mutant_data)), list_size))
        else: # numerous combinations, random select instead
            self.comb_idx = None
    
    def __len__(self):
        if self.comb_idx is not None:
            return len(self.comb_idx)
        else:
            return self.max_size
    
    def __getitem__(self, idx): # yield combination indices instead of real data
        if self.comb_idx is not None:
            comb =  self.comb_idx[idx]
        else:
            comb =  random.sample(range(len(self.mutant_data)), self.list_size)

        # 如果是主动学习阶段且存在偏好对，返回偏好对索引
        if self.active_learning and self.current_pref_pairs is not None:
            return (*comb, self.current_pref_pairs)
        else:
            return comb

    def update_pref_pairs(self, pref_pairs):
        """用于主动学习阶段更新偏好对"""
        self.current_pref_pairs = pref_pairs


    def collate(self, batch_with_pref): # identify duplicate elements among a batch of combinations
        # 分离组合索引和偏好对
        if self.active_learning and self.current_pref_pairs is not None:
            comb_idx = [item[:-1] for item in batch_with_pref]
            pref_pairs = batch_with_pref[0][-1]  # 所有样本共享同一批偏好对
        else:
            comb_idx = batch_with_pref
            pref_pairs = None
        comb_idx = torch.tensor(comb_idx, device=self.device)
        unique_mt, inv_idx = torch.unique(comb_idx, return_inverse=True)
        raw_batch = [self.mutant_data[i] for i in unique_mt]
        output_batch = self.mutant_data.collate(raw_batch)
        output_batch['inv_list_idx'] = inv_idx
        if pref_pairs is not None:
            output_batch['pref_pairs'] = pref_pairs
        return output_batch


def generate_pref_pairs_from_scores(scores, batch_size=12,
                                    high_diff_ratio=0.8,
                                    top_score_ratio=0.3,
                                    current_epoch=0,
                                    total_epochs=10):
    device = scores.device
    n = len(scores)
    pref_pairs = []

    # 动态调整策略参数
    dynamic_ratio = 0.5 + 0.5 * (current_epoch / total_epochs)
    high_diff_num = int(batch_size * high_diff_ratio * dynamic_ratio)
    topk_num = max(1, int(n * top_score_ratio * dynamic_ratio))

    # ===== 阶段1：高分区差异对比 =====
    if topk_num > 0:
        topk_scores, topk_indices = scores.topk(topk_num)
        candidate_pool = topk_indices

        if high_diff_num > 0 and len(candidate_pool) > 1:
            # 确保索引张量在相同设备
            candidate_scores = scores[candidate_pool]

            # 构建三角差异矩阵
            idx_i, idx_j = torch.combinations(torch.arange(len(candidate_pool), device=device), 2).unbind(1)
            diff = candidate_scores[idx_i] - candidate_scores[idx_j]

            # 计算合理差异范围
            abs_diff = diff.abs()
            median_diff = torch.median(abs_diff)
            if not torch.isnan(median_diff):
                valid_mask = (abs_diff > median_diff * 0.5) & (abs_diff < median_diff * 2)

                if valid_mask.any():
                    valid_i = idx_i[valid_mask]
                    valid_j = idx_j[valid_mask]

                    # 强制确保正确顺序
                    swap_mask = candidate_scores[valid_i] < candidate_scores[valid_j]
                    valid_i[swap_mask], valid_j[swap_mask] = valid_j[swap_mask], valid_i[swap_mask]

                    # 随机选择
                    num_valid = len(valid_i)
                    selected = torch.randperm(num_valid, device=device)[:min(high_diff_num, num_valid)]

                    for i in selected:
                        pref_pairs.append((
                            candidate_pool[valid_i[i]],
                            candidate_pool[valid_j[i]]
                        ))

    # ===== 阶段2：高低分区严格对比 =====
    remaining_num = batch_size - len(pref_pairs)
    if remaining_num > 0:
        # 处理没有候选池的情况
        if topk_num == 0 or len(candidate_pool) == 0:
            # 创建简单对比对
            indices = torch.randperm(n, device=device)[:2 * remaining_num]
            indices = indices.reshape(-1, 2)
            for i, j in indices:
                if scores[i] > scores[j]:
                    pref_pairs.append((i, j))
                else:
                    pref_pairs.append((j, i))
            return torch.stack(pref_pairs)

        # 创建非高分候选池
        all_indices = torch.arange(n, device=device)
        # 使用GPU兼容的方法创建掩码
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        mask[candidate_pool] = True
        non_candidate_pool = all_indices[~mask]

        if len(non_candidate_pool) > 0:
            # 从高低分区采样
            high_idx = torch.randint(0, len(candidate_pool), (remaining_num,), device=device)
            low_idx = torch.randint(0, len(non_candidate_pool), (remaining_num,), device=device)

            for h, l in zip(high_idx, low_idx):
                i = candidate_pool[h]
                j = non_candidate_pool[l]
                # 确保顺序
                if scores[i] > scores[j]:
                    pref_pairs.append((i, j))
                else:
                    pref_pairs.append((j, i))
        else:
            # 所有样本都是高分样本时
            idx_i, idx_j = torch.combinations(candidate_pool, 2).unbind(1)
            selected = torch.randperm(len(idx_i), device=device)[:remaining_num]
            for idx in selected:
                i, j = idx_i[idx], idx_j[idx]
                if scores[i] > scores[j]:
                    pref_pairs.append((i, j))
                else:
                    pref_pairs.append((j, i))

    return torch.tensor(pref_pairs, device=device)

def generate_pref_pairs_from_model(model, dataset, k=20):
    """主动学习阶段：选择模型预测不确定的高分序列对"""
    with torch.no_grad():
        all_scores = []
        for batch in DataLoader(dataset, batch_size=32):
            all_scores.append(model.predict(batch['sequences']))
        rewards = torch.cat(all_scores).squeeze()
        # 选择高分且预测接近的序列对
    high_score_idx = torch.topk(rewards, k * 2).indices
    candidate_pairs = torch.combinations(high_score_idx, r=2)
    pair_diff = torch.abs(rewards[candidate_pairs[:, 0]] - rewards[candidate_pairs[:, 1]])
    selected_pairs = candidate_pairs[pair_diff.topk(k, largest=False).indices]

    return selected_pairs