import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3
num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3 

class GINEConv(MessagePassing):  # 添加自环，编码边特征
    def __init__(self, emb_dim):  # 300
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(  # 序列容器--因该是每一层GCN后面接一层这个
            nn.Linear(emb_dim, 2*emb_dim),  # 300 600
            nn.ReLU(), 
            nn.Linear(2*emb_dim, emb_dim)  # 600  300
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)  # 编码键类型 5 300
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)  # 编码键方向 3 300
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):  # n*2  2*2e  2e*2--n节点数 e边数
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 # bond type for self-loop edge。 键方向默认0，表示无。
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # 键的两部分特征编码后组合  dim=600
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # 将(添加自环的邻接矩阵  原始节点特征  编码好的边特征)输入到self.propagate进行学习
        # self.propagate继承自pyg类中的函数，包含GNN网络操作。
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)  # dim=300

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool='mean'):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        # nn.Embedding(a,emb_dim）根据原子编号进行编码扩充--第一个参数a编码范围从0开始计算
        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)  # 119  300
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)  # 3  300
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)  # 均匀分布初始化网络参数
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):  # 5
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))  # 300
            # nn.BatchNorm1d(dim)，dim等于前一层输出的维度。BatchNorm层输出的维度也是dim。

        if pool == 'mean':  #
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)  # 300  256

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),   # 256  256
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim//2)  # 256  128
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])  # 节点特征嵌入编码

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)  # 返回gnn学习结果,dim=300，边特征的编码包含其中
            h = self.batch_norms[layer](h)  # 归一化
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        out = self.out_lin(h)
        
        return h, out  # 表示(下游任务)：256  投影（对比损失）：128
