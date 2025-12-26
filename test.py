def train_one_epoch(self, dataloader):
    self.model.train()
    total_loss, ce_acc, nce_acc = 0.0, 0.0, 0.0
    for x, y in dataloader:
        x, y = x.to(self.device), y.to(self.device)
        batch_size = x.size(0)
        # --- Task 1: Main Classification Task ---
        logits_clean, _ = self.model(x)
        loss_ce = F.cross_entropy(logits_clean, y)
        # --- Task 2: Label-Aware Self-Supervised Task ---
        # 1. Augmentation
        x_view1 = self.augmenter(x)
        x_view2 = self.augmenter(x)
        # 2. Projection
        _, proj1 = self.model(x_view1)
        _, proj2 = self.model(x_view2)
        features = torch.cat([proj1, proj2], dim=0)  # [2B, D]
        # 3. Similarity Matrix
        sim_matrix = torch.matmul(features, features.T) / self.kwargs['temperature']
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()  # Stability
        # 4. 构建 Mask (创新点在这里)
        # Label Mask: 标记出 Batch 中 label 相同的样本对
        # y_cat: [y, y] -> [2B]
        y_cat = torch.cat([y, y], dim=0)
        # mask_label[i, j] = 1 if label[i] == label[j]
        mask_label = torch.eq(y_cat.unsqueeze(0), y_cat.unsqueeze(1)).float()
        # Identity Mask: 排除自身 (对角线)
        mask_self = torch.eye(2 * batch_size, device=self.device)
        # Positive Mask: 仅指 SimCLR 定义的正样本 (i, i+B) 和 (i+B, i)
        # 这是我们想要在分子中拉近的目标
        mask_pos = torch.zeros((2 * batch_size, 2 * batch_size), device=self.device)
        mask_pos[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = 1
        mask_pos[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = 1
        # Negative Mask (分母的 Mask):
        # 原始 SimCLR: 只要不是自己，都是负样本 -> (1 - mask_self)
        # 创新改进 FN-Cancellation: 只要不是自己，且 *label不同*，才是负样本
        # 也就是说：如果 label 相同 (mask_label==1)，我们既不把它当正样本(分子)，也不把它当负样本(分母)，直接忽略
        # 逻辑：
        # 真正的负样本 = (Label不同) OR (是自身对应的增强视图-即原本的正样本对)
        # 注意：SimCLR公式中分母通常包含正样本项，为了标准实现，我们通常保留正样本在分母，剔除其他同label的项
        # 定义：允许在分母中出现的项 = (Label不同的项) + (SimCLR定义的正样本对)
        mask_valid_neg = (1 - mask_label) + mask_pos
        # 确保对角线不参与
        mask_valid_neg = mask_valid_neg * (1 - mask_self)
        # 限制为 0/1
        mask_valid_neg = (mask_valid_neg > 0).float()
        # 5. Compute Loss
        exp_sim = torch.exp(sim_matrix)
        # 分子: exp(sim(pos))
        pos_sim = (exp_sim * mask_pos).sum(dim=1)
        # 分母: sum(exp(sim(negatives)))
        # 只累加 valid_neg mask 为 1 的部分
        neg_sim_sum = (exp_sim * mask_valid_neg).sum(dim=1)
        # Log Prob
        # loss = -log ( pos / (pos + negs) )
        # 注意：SimCLR 标准公式分母包含正样本项。
        # mask_valid_neg 包含了 mask_pos，所以 neg_sim_sum 已经包含了 pos_sim
        log_prob = -torch.log(pos_sim / (neg_sim_sum + 1e-8) + 1e-8)
        loss_nce = log_prob.mean()
        # Total Loss
        loss = loss_ce + self.kwargs['lambda_nce'] * loss_nce
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        total_loss += loss.item()
        ce_acc += loss_ce.item()
        nce_acc += loss_nce.item()
    n = len(dataloader)
    return total_loss / n, ce_acc / n, nce_acc / n