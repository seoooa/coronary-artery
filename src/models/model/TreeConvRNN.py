import torch as th
import torch.nn as nn
import dgl
from dgl.data.utils import load_graphs
import networkx as nx

'''
Learning tree-structured representation for 3d coronary artery segmentation
'''
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv3d(1, 16, 3, 1, 1), nn.ReLU(), nn.Conv3d(16, 16, 3, 1, 1), nn.ReLU())
        self.conv_2 = nn.Sequential(nn.Conv3d(16, 32, 3, 1, 1), nn.ReLU(), nn.Conv3d(32, 20, 3, 1, 1), nn.ReLU())

        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool3d(2, 2)
        self.pooling2 = nn.MaxPool3d(2, 2)

    def forward(self, inputs):
        x0 = self.conv_1(inputs)
        x = self.pooling(x0)
        x1 = self.conv_2(x)
        x = self.pooling2(x1)
        return x0, x1, x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(48, 16, 3, 1, 1), nn.Dropout(), nn.ReLU(), nn.Conv3d(16, 16, 3, 1, 1),
                                   nn.Dropout(), nn.ReLU(), nn.Conv3d(16, 1, 1, 1, 0))
        self.conv2 = nn.Sequential(nn.Conv3d(30, 20, 3, 1, 1), nn.ReLU(), nn.Conv3d(20, 32, 3, 1, 1), nn.ReLU())
        self.relu = nn.ReLU()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, inputs, x0, x1):
        x = self.up(inputs)
        x = th.cat([x, x1], dim=1)
        x = self.conv2(x)
        x = self.up(x)
        x = th.cat([x, x0], dim=1)
        x = self.conv1(x)
        return x


# 实现treeConvLSTM,treeConvGRU
class TreeConvLSTMCell3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, h_list):
        super(TreeConvLSTMCell3d, self).__init__()
        self.W_iou = nn.Sequential(nn.Conv3d(in_channel, 3 * out_channel, kernel_size, 1, 1, bias=False))
        self.U_iou = nn.Conv3d(out_channel, 3 * out_channel, kernel_size, 1, 1, bias=False)
        self.b_iou = nn.Parameter(th.zeros(3 * out_channel, h_list[0] // 4, h_list[1] // 4, h_list[2] // 4))
        self.U_f = nn.Conv3d(out_channel, out_channel, kernel_size, 1, 1, bias=False)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        # 假设有n个node,nodes.mailbox['h'] shape=[相同入度的节点个数,n,channel,H,W],h_child [相同入度的节点个数,channel,H,W]
        h_child = th.sum(nodes.mailbox['h'], 1)
        #
        s1, s2, s3, s4, s5, s6 = nodes.mailbox['h'].size()
        f = th.sigmoid(self.U_f(nodes.mailbox['h'].view(s1 * s2, s3, s4, s5, s6)))
        f = f.view(s1, s2, s3, s4, s5, s6)
        # f=th.sigmoid(self.U_f(nodes.mailbox['h'].view()))

        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_child), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        # print(i.size())
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)

        return {'h': h, 'c': c}


class TreeConvLSTM3d(nn.Module):
    def __init__(self, input_channel, output_channel, h_list):
        super(TreeConvLSTM3d, self).__init__()

        cell = TreeConvLSTMCell3d
        self.cell = cell(input_channel, output_channel, 3, h_list)
        # self.conv = nn.Conv3d(output_channel, 2, 3, 1, 1)
        self.encoder = Encoder()
        self.decoder = Decoder()
        # self.dgc = DGCNet(input_channel, input_channel // 2, input_channel // 4)
        self.act_fun = nn.Sigmoid()

    def forward(self, g, h, c):
        # g = batch.graph
        x0, x1, x = self.encoder(g.ndata['data'])
        # x=self.dgc(x)
        g.ndata['iou'] = self.cell.W_iou(x)
        g.ndata['h'] = h
        g.ndata['c'] = c
        # 传播消息
        # dgl.prop_nodes_topo(g)
        dgl.prop_nodes_topo(g, self.cell.message_func, self.cell.reduce_func, apply_node_func=self.cell.apply_node_func)
        # 这里需要一个解码网络
        h = g.ndata.pop('h')
        h = self.decoder(h, x0, x1)
        logits = h
        return logits


class TreeConvGRUCell3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(TreeConvGRUCell3d, self).__init__()
        self.W = nn.Conv3d(in_channel, out_channel, kernel_size, 1, 1, bias=False)
        self.W_z = nn.Conv3d(in_channel, out_channel, kernel_size, 1, 1, bias=False)
        self.W_r = nn.Conv3d(in_channel, out_channel, kernel_size, 1, 1, bias=False)
        self.U_z = nn.Conv3d(out_channel, out_channel, kernel_size, 1, 1)
        self.U_r = nn.Conv3d(out_channel, out_channel, kernel_size, 1, 1)
        self.U = nn.Conv3d(out_channel, out_channel, kernel_size, 1, 1)

    def message_func(self, edges):
        return {'h': edges.src['h']}

    def reduce_func(self, nodes):
        # 假设有n个node,nodes.mailbox['h'] shape=[相同入度的节点个数,n,channel,H,W],h_child [相同入度的节点个数,channel,H,W]
        h_child = th.sum(nodes.mailbox['h'], 1)
        # nodes_num = h_child.size()[0]
        # rjk = th.zeros_like(nodes.mailbox['h'])
        # h_k = th.zeros_like(nodes.mailbox['h'])
        # for i in range(nodes_num):
        #     rjk[i, :, :, :, :] = self.U_r(nodes.mailbox['h'][i, :, :, :, :])
        #     h_k[i, :, :, :, :]=self.U(nodes.mailbox['h'][i, :, :, :, :])

        s1, s2, s3, s4, s5, s6 = nodes.mailbox['h'].size()
        rjk = self.U_r(nodes.mailbox['h'].view(s1 * s2, s3, s4, s5, s6))
        h_k = self.U(nodes.mailbox['h'].view(s1 * s2, s3, s4, s5, s6))
        w = th.sum(th.sigmoid(rjk + nodes.data['r'].unsqueeze(dim=1)) * h_k, 1) + nodes.data['w']
        u = self.U_z(h_child) + nodes.data['u']
        return {'u': u, 'h': h_child, 'w': w}

    def apply_node_func(self, nodes):
        w = nodes.data['w']
        u = nodes.data['u']
        u, w = th.sigmoid(u), th.tanh(w)
        h = u * nodes.data['h'] + (1 - u) * w
        return {'h': h}


class TreeConvGRU3d(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(TreeConvGRU3d, self).__init__()
        cell = TreeConvGRUCell3d
        self.cell = cell(input_channel, output_channel, 3)
        # self.conv = nn.Conv2d(output_channel, 2, 3, 1, 1)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.act_fun = nn.Sigmoid()

    def forward(self, g, h):
        x0, x1, x = self.encoder(g.ndata['data'])
        # x=self.dgc(x)
        g.ndata['w'] = self.cell.W(x)
        g.ndata['u'] = self.cell.W_z(x)
        g.ndata['r'] = self.cell.W_r(x)
        g.ndata['h'] = h
        # 传播消息
        # dgl.prop_nodes_topo(g)
        dgl.prop_nodes_topo(g, self.cell.message_func, self.cell.reduce_func, apply_node_func=self.cell.apply_node_func)
        # 这里需要一个解码网络
        h = g.ndata.pop('h')
        h = self.decoder(h, x0, x1)
        logits = h
        return logits

#
# def test_GRU():
#     device = th.device('cuda' if th.cuda.is_available() else 'cpu')
#     g_load = load_graphs('g1.bin')
#     g = g_load[0]
#     g = g[0]
#     g.ndata['img'] = g.ndata['img'].float()
#     g.ndata['img'] = g.ndata['img'].to(device)
#     net = TreeConvGRU3d(20, 10).to(device)
#     batch_g = dgl.batch([g])
#     n = g.number_of_nodes()
#     print(n)
#     h = th.zeros((n, 10, 16, 16)).to(device)
#     c = th.zeros((n, 10, 16, 16)).to(device)
#     output = net(g, h)
