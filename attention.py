import torch
import torch.nn as nn
import argparse

p = argparse.ArgumentParser()
p.add_argument('--hid_dim', type=int, default=132, help='node, edge, fg hidden dims in Net')
p.add_argument('--heads', type=int, default=4, help='Multi-head num')
args = p.parse_args()
class LocalAugmentation(nn.Module):
    def __init__(self):
        super(LocalAugmentation, self).__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(args.hid_dim, args.hid_dim, bias=False) for _ in range(3)])
        self.W_o = nn.Linear(args.hid_dim, args.hid_dim)
        self.heads = args.heads  # args.heads=4
        self.d_k = args.hid_dim // args.heads  # 33

    def forward(self, knowledge1, knowledge2, knowledge3):
        batch_size = knowledge1.shape[0]
        hid_dim = knowledge1.shape[-1]

        Q = knowledge3
        Q = Q.view(batch_size, -1, 1, hid_dim).transpose(1, 2)
        K = []
        K.append(knowledge1.unsqueeze(1))
        K.append(knowledge2.unsqueeze(1))
        K = torch.cat(K, dim=1)
        K = K.view(batch_size, -1, 1, hid_dim).transpose(1, 2)
        V = K
        Q, K, V = [l(x).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linear_layers, (Q, K, V))]
        message_interaction = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k
        att_score = torch.nn.functional.softmax(message_interaction, dim=-1)
        motif_messages = torch.matmul(att_score, V).transpose(1, 2).contiguous().view(batch_size, -1, hid_dim)
        motif_messages = self.W_o(motif_messages)
        return motif_messages.squeeze(1)
