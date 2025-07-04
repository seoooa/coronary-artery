import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, SAGEConv
from dgl.data.utils import load_graphs


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.act = nn.Tanh()
        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, activation=nn.ReLU()))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, activation=nn.ReLU()))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type))  # activation None

    def forward(self, graph, inputs):
        # h = self.dropout(inputs)
        h = inputs
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.dropout(h)
        return h


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        self.l2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.l3 = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.l4 = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.l5 = nn.Sequential(nn.Linear(64, 1))

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        return x

# if __name__ == '__main__':
#     g = load_graphs('/home/wcb/Python_code/Paper_implementation/Paper_4/graph_data/32_dim/train/12021822_1.bin', [0])[0][0]
#     print(g)
#     print(torch.max(g.ndata['rv']))
#     # # net=GCN(g,32,64,1,4,None,0.5)
#     # # net=MLP()
#     net = GraphSAGE(32, 64, 1, 4, None, 0.5, aggregator_type='gcn')
#
#     print(get_parameter_number(net))
#     # # print(g.ndata['xv'].shape)
#     # y=net(g,g.ndata['xv'].float())
#     # print(y.shape)
#     # print(y)
#     # # pass
