import torch
import numpy as np


class NTXentLoss(torch.nn.Module):   # 计算相关损失--投影计算对比损失，表示用于下游任务

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature          # 0.1
        self.device = device

        self.softmax = torch.nn.Softmax(dim=-1)  # 相当于每行求
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)  # 掩码矩阵true/false
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)  # true，返回计算相似性的函数
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        # 当reduction='sum'时，输出是对这一个batch预测的损失之和

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)  # 权重相似度，两个向量按照行求相似度
            return self._cosine_simililarity
        else:
            return self._dot_simililarity  # 普通内积

    def _cosine_simililarity(self, x, y):
        # x shape: (2B, 1, D)--锚点分子
        # y shape: (1, D, 2B)--增强样本
        # v shape: (2B, 2B)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))  # unsqueeze(dim)扩充对应维度
        return v  # 锚点分子与每个增强样本的相似度

    def _get_correlated_mask(self):
        # np.eye（N, M, k）--返回的是一个二维2的数组(N,M)，对角线的地方为1，其余的地方为0
        # N:表示的是输出的行数M：
        # M: 可选项，输出的列数，如果没有就默认为N
        # k：int型，可选项，对角线的下标，默认为0表示的是主对角线，负数表示的是低对角，正数表示的是高对角。
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)  # l1 l2 是对称的
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))  # 组合三部分矩阵，值 0 1
        mask = (1 - mask).type(torch.bool)  # 构成掩码矩阵，值true false
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):  # 类似矩阵做内积
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (2B, 1, D)--锚点分子
        # y shape: (1, D, 2B)--增强样本
        # v shape: (2B, 2B)
        return v

    def forward(self, z_ori, z_hard_pos, z_soft_pos, z_soft_neg):  # 直接拼接求相似度计算损失

        representations_ori_hp = torch.cat([z_ori, z_hard_pos], dim=0)  # B+B x D
        representations_ori_sp = torch.cat([z_ori, z_soft_pos], dim=0)  # B+B x D
        representations_ori_sn = torch.cat([z_ori, z_soft_neg], dim=0)  # B+B x D
        representations_hp_sp = torch.cat([z_hard_pos, z_soft_pos], dim=0)  # B+B x D
        representations_hp_sn = torch.cat([z_hard_pos, z_soft_neg], dim=0)  # B+B x D
        representations_sp_sn = torch.cat([z_soft_pos, z_soft_neg], dim=0)  # B+B x D

        # 求 余弦相似度  即 sim 函数。
        similarity_matrix_ori_hp = self.similarity_function(representations_ori_hp, representations_ori_hp)     # 2B x 2B
        similarity_matrix_ori_sp = self.similarity_function(representations_ori_sp, representations_ori_sp)     # 2B x 2B
        similarity_matrix_ori_sn = self.similarity_function(representations_ori_sn, representations_ori_sn)     # 2B x 2B
        similarity_matrix_hp_sp = self.similarity_function(representations_hp_sp, representations_hp_sp)        # 2B x 2B
        similarity_matrix_hp_sn = self.similarity_function(representations_hp_sn, representations_hp_sn)        # 2B x 2B
        similarity_matrix_sp_sn = self.similarity_function(representations_sp_sn, representations_sp_sn)        # 2B x 2B

        l_pos_ori_sn = torch.diag(similarity_matrix_ori_sn, self.batch_size)    # right positive
        r_pos_ori_sn = torch.diag(similarity_matrix_ori_sn, -self.batch_size)   # left positive
        positives_ori_sn = torch.cat([l_pos_ori_sn, r_pos_ori_sn]).view(2 * self.batch_size, 1)

        l_pos_hp_sn = torch.diag(similarity_matrix_hp_sn, self.batch_size)    # right positive
        r_pos_hp_sn = torch.diag(similarity_matrix_hp_sn, -self.batch_size)   # left positive
        positives_hp_sn = torch.cat([l_pos_hp_sn, r_pos_hp_sn]).view(2 * self.batch_size, 1)

        l_pos_sp_sn = torch.diag(similarity_matrix_sp_sn, self.batch_size)    # right positive
        r_pos_sp_sn = torch.diag(similarity_matrix_sp_sn, -self.batch_size)   # left positive
        positives_sp_sn = torch.cat([l_pos_sp_sn, r_pos_sp_sn]).view(2 * self.batch_size, 1)
        # negatives_ori_sn = similarity_matrix_ori_sn[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        # filter out the scores from the positive samples
        l_pos_ori_hp = torch.diag(similarity_matrix_ori_hp, self.batch_size)    # right positive
        r_pos_ori_hp = torch.diag(similarity_matrix_ori_hp, -self.batch_size)   # left positive
        positives_ori_hp = torch.cat([l_pos_ori_hp, r_pos_ori_hp]).view(2 * self.batch_size, 1)
        negatives_ori_hp = similarity_matrix_ori_hp[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        l_pos_ori_sp = torch.diag(similarity_matrix_ori_sp, self.batch_size)
        r_pos_ori_sp = torch.diag(similarity_matrix_ori_sp, -self.batch_size)
        positives_ori_sp = torch.cat([l_pos_ori_sp, r_pos_ori_sp]).view(2 * self.batch_size, 1)     # batch * 1
        negatives_ori_sp = similarity_matrix_ori_sp[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        l_pos_hp_sp = torch.diag(similarity_matrix_hp_sp, self.batch_size)
        r_pos_hp_sp = torch.diag(similarity_matrix_hp_sp, -self.batch_size)
        positives_hp_sp = torch.cat([l_pos_hp_sp, r_pos_hp_sp]).view(2 * self.batch_size, 1)     # batch * 1
        negatives_hp_sp = similarity_matrix_hp_sp[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits_ori_hp = torch.cat((positives_ori_hp, negatives_ori_hp, positives_ori_sn), dim=1)   # batch * 3
        logits_ori_hp /= self.temperature

        logits_ori_sp = torch.cat((positives_ori_sp, negatives_ori_sp, positives_ori_sn), dim=1)   # batch * 3
        logits_ori_sp /= self.temperature

        logits_hp_sp = torch.cat((positives_hp_sp, negatives_hp_sp, positives_hp_sn, positives_sp_sn), dim=1)   # batch * 3
        logits_hp_sp /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()    # 2 * batch
        loss_ori_hp = self.criterion(logits_ori_hp, labels)
        loss_ori_sp = self.criterion(logits_ori_sp, labels)
        logits_hp_sp = self.criterion(logits_hp_sp, labels)

        return loss_ori_hp / (2 * self.batch_size), loss_ori_sp / (2 * self.batch_size), logits_hp_sp / (2 * self.batch_size)


class Weight_NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(Weight_NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature          # 0.1
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)     # true
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (2B, 1, D)
        # y shape: (1, D, 2B)
        # v shape: (2B, 2B)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (2B, 1, D)
        # y shape: (1, D, 2B)
        # v shape: (2B, 2B)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, z_ori, z_hard_pos, z_soft_pos, z_soft_neg):

        # torch.stack默认dim=0，把2个2维的张量凑成一个3维的张量：2*B*D；torch.mean求得均值后会变为两维:B*D
        # 下面是对batch数据进行预处理，求得两个向量对应位置两个值的均值--如锚点和硬正每个位置两个数的均值
        representations_ori_hp_averge = torch.mean(torch.stack((z_ori, z_hard_pos)), dim=0)   # Bx D  对三维矩阵的dim=0求平均，
        representations_ori_sp_averge = torch.mean(torch.stack((z_ori, z_soft_pos)), dim=0)   # B x D
        representations_ori_sn_averge = torch.mean(torch.stack((z_ori, z_soft_neg)), dim=0)   # B x D
        representations_hp_sp_averge = torch.mean(torch.stack((z_hard_pos, z_soft_pos)), dim=0)   # Bx D
        representations_hp_sn_averge = torch.mean(torch.stack((z_hard_pos, z_soft_neg)), dim=0)   # B x D
        representations_sp_sn_averge = torch.mean(torch.stack((z_soft_pos, z_soft_neg)), dim=0)   # Bx D

        #  下面操作，原始数据减去相应均值，然后按照列(下)拼接，维度2B*D
        representations_ori_hp = torch.cat([z_ori-representations_ori_hp_averge, z_hard_pos-representations_ori_hp_averge], dim=0)  # 锚点-硬正 B+B x D
        representations_ori_sp = torch.cat([z_ori-representations_ori_sp_averge, z_soft_pos-representations_ori_sp_averge], dim=0)  # 锚点-软正 B+B x D
        representations_ori_sn = torch.cat([z_ori-representations_ori_sn_averge, z_soft_neg-representations_ori_sn_averge], dim=0)  # 锚点-软负 B+B x D
        representations_hp_sp = torch.cat([z_hard_pos-representations_hp_sp_averge, z_soft_pos-representations_hp_sp_averge], dim=0)  # 硬正-软正 B+B x D
        representations_hp_sn = torch.cat([z_hard_pos-representations_hp_sn_averge, z_soft_neg-representations_hp_sn_averge], dim=0)  # 硬正-软负 B+B x D
        representations_sp_sn = torch.cat([z_soft_pos-representations_sp_sn_averge, z_soft_neg-representations_sp_sn_averge], dim=0)  # 软正-软负 B+B x D

        # 求 弦相似度，得到相似度矩阵。 self.similarity_function（x, y, v）
        # x shape: (2B, 1, D)
        # y shape: (1, D, 2B)
        # v shape: (2B, 2B)
        similarity_matrix_ori_hp = self.similarity_function(representations_ori_hp, representations_ori_hp)     # 2B x 2B
        similarity_matrix_ori_sp = self.similarity_function(representations_ori_sp, representations_ori_sp)     # 2B x 2B
        similarity_matrix_ori_sn = self.similarity_function(representations_ori_sn, representations_ori_sn)     # 2B x 2B
        similarity_matrix_hp_sp = self.similarity_function(representations_hp_sp, representations_hp_sp)        # 2B x 2B
        similarity_matrix_hp_sn = self.similarity_function(representations_hp_sn, representations_hp_sn)        # 2B x 2B
        similarity_matrix_sp_sn = self.similarity_function(representations_sp_sn, representations_sp_sn)        # 2B x 2B


        # 锚点-软负
        l_pos_ori_sn = torch.diag(similarity_matrix_ori_sn, self.batch_size)    # torch.diag（）取矩阵的对角线元素。right positive
        r_pos_ori_sn = torch.diag(similarity_matrix_ori_sn, -self.batch_size)   # left positive
        positives_ori_sn = torch.cat([l_pos_ori_sn, r_pos_ori_sn]).view(2 * self.batch_size, 1)  # 拼接并转换维度--维度为什么是2B*1??

        # 硬正-软负
        l_pos_hp_sn = torch.diag(similarity_matrix_hp_sn, self.batch_size)    # right positive
        r_pos_hp_sn = torch.diag(similarity_matrix_hp_sn, -self.batch_size)   # left positive
        positives_hp_sn = torch.cat([l_pos_hp_sn, r_pos_hp_sn]).view(2 * self.batch_size, 1)

        # 软正-软负
        l_pos_sp_sn = torch.diag(similarity_matrix_sp_sn, self.batch_size)    # right positive
        r_pos_sp_sn = torch.diag(similarity_matrix_sp_sn, -self.batch_size)   # left positive
        positives_sp_sn = torch.cat([l_pos_sp_sn, r_pos_sp_sn]).view(2 * self.batch_size, 1)
        # negatives_ori_sn = similarity_matrix_ori_sn[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        # 锚点-硬正  filter out the scores from the positive samples
        l_pos_ori_hp = torch.diag(similarity_matrix_ori_hp, self.batch_size)    # right positive
        r_pos_ori_hp = torch.diag(similarity_matrix_ori_hp, -self.batch_size)   # left positive
        positives_ori_hp = torch.cat([l_pos_ori_hp, r_pos_ori_hp]).view(2 * self.batch_size, 1)
        negatives_ori_hp = similarity_matrix_ori_hp[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)  # 通过掩码矩阵得到负的？？

        # 锚点-软正
        l_pos_ori_sp = torch.diag(similarity_matrix_ori_sp, self.batch_size)
        r_pos_ori_sp = torch.diag(similarity_matrix_ori_sp, -self.batch_size)
        positives_ori_sp = torch.cat([l_pos_ori_sp, r_pos_ori_sp]).view(2 * self.batch_size, 1)     # batch * 1
        negatives_ori_sp = similarity_matrix_ori_sp[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        # 硬正-软正
        l_pos_hp_sp = torch.diag(similarity_matrix_hp_sp, self.batch_size)
        r_pos_hp_sp = torch.diag(similarity_matrix_hp_sp, -self.batch_size)
        positives_hp_sp = torch.cat([l_pos_hp_sp, r_pos_hp_sp]).view(2 * self.batch_size, 1)     # batch * 1
        negatives_hp_sp = similarity_matrix_hp_sp[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)


        # ?????????????
        logits_ori_hp = torch.cat((positives_ori_hp, negatives_ori_hp, positives_ori_sn), dim=1)   # batch * 3
        logits_ori_hp /= self.temperature

        logits_ori_sp = torch.cat((positives_ori_sp, negatives_ori_sp, positives_ori_sn), dim=1)   # batch * 3
        logits_ori_sp /= self.temperature

        logits_hp_sp = torch.cat((positives_hp_sp, negatives_hp_sp, positives_hp_sn, positives_sp_sn), dim=1)   # batch * 3
        logits_hp_sp /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()    # 2 * batch
        loss_ori_hp = self.criterion(logits_ori_hp, labels)
        loss_ori_sp = self.criterion(logits_ori_sp, labels)
        logits_hp_sp = self.criterion(logits_hp_sp, labels)

        return loss_ori_hp / (2 * self.batch_size), loss_ori_sp / (2 * self.batch_size), logits_hp_sp / (2 * self.batch_size)
