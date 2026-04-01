import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2 * cheb_k * dim_in, dim_out))  # 2 is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, supports):
        x_g = []
        support_set = []
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
            support_set.extend(support_ks)
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1)  # B, N, 2 * cheb_k * dim_in
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k)
        self.update = AGCN(dim_in + self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, supports):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class ADCRNN_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(ADCRNN_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, x, init_state, supports):
        # shape of x: (B, T, N, D), shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, supports)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states


class ADCRNN_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(ADCRNN_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, xt, init_state, supports):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class MegaCRNForPowerPred(nn.Module):
    """适配电力预测任务的MegaCRN，输入格式 [batch, sites, timesteps, features]，输出 [batch, sites, pred_steps]"""

    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, num_layers=1, cheb_k=3,
                 mem_num=20, mem_dim=64, use_curriculum_learning=True, cl_decay_steps=2000):
        super(MegaCRNForPowerPred, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.rnn_units = rnn_units
        self.output_dim = output_dim  # 预测每个时间步的功率，通常为1
        self.horizon = horizon  # 预测步长
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning

        # memory
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.memory = self.construct_memory()

        # encoder
        self.encoder = ADCRNN_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.num_layers)

        # decoder - 注意这里去掉了ycov_dim
        self.decoder_dim = self.rnn_units + self.mem_dim
        # 解码器输入维度只包含预测值，不需要协变量
        self.decoder = ADCRNN_Decoder(self.num_nodes, self.output_dim, self.decoder_dim, self.cheb_k, self.num_layers)

        # output
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))

        # 可选的：为解码器提供时间特征嵌入
        self.time_embedding = nn.Embedding(horizon, 1)  # 简单的时间位置编码

    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)  # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim),
                                         requires_grad=True)  # project to query
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num),
                                          requires_grad=True)  # project memory to embedding
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num),
                                          requires_grad=True)  # project memory to embedding
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def query_memory(self, h_t: torch.Tensor):
        query = torch.matmul(h_t, self.memory['Wq'])  # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)  # alpha: (B, N, M)
        value = torch.matmul(att_score, self.memory['Memory'])  # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.memory['Memory'][ind[:, :, 0]]  # B, N, d
        neg = self.memory['Memory'][ind[:, :, 1]]  # B, N, d
        return value, query, pos, neg

    def forward(self, x, labels=None, batches_seen=None):
        """
        输入 x: [batch_size, num_sites, in_steps, input_dim]
        输出: [batch_size, num_sites, horizon]
        """
        # 调整输入维度: [B, N, T, F] -> [B, T, N, F]
        x = x.permute(0, 2, 1, 3)  # [batch_size, in_steps, num_nodes, input_dim]
        batch_size, in_steps, num_nodes, input_dim = x.shape

        # 生成自适应图
        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        supports = [g1, g2]

        # 编码器
        init_state = self.encoder.init_hidden(batch_size)
        h_en, state_en = self.encoder(x, init_state, supports)  # B, T, N, hidden
        h_t = h_en[:, -1, :, :]  # B, N, hidden (last state)

        # 记忆查询
        h_att, query, pos, neg = self.query_memory(h_t)
        h_t = torch.cat([h_t, h_att], dim=-1)  # B, N, decoder_dim

        # 解码器
        ht_list = [h_t] * self.num_layers
        go = torch.zeros((batch_size, self.num_nodes, self.output_dim), device=x.device)
        out = []

        for t in range(self.horizon):
            # 使用时间嵌入作为额外的输入特征（可选）
            time_emb = self.time_embedding(torch.tensor([t], device=x.device)).expand(batch_size, num_nodes, 1)
            decoder_input = go + time_emb  # 添加时间信息

            h_de, ht_list = self.decoder(decoder_input, ht_list, supports)
            go = self.proj(h_de)
            out.append(go)

            # 课程学习（如果提供标签）
            if self.training and self.use_curriculum_learning and labels is not None:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    # 注意：这里labels需要是 [B, N, T] 格式
                    if len(labels.shape) == 4:  # [B, N, T, 1]
                        go = labels[:, :, t, :]
                    else:  # [B, N, T]
                        go = labels[:, :, t].unsqueeze(-1)

        output = torch.stack(out, dim=2)  # [B, N, horizon, output_dim]
        output = output.squeeze(-1)  # [B, N, horizon] if output_dim=1

        return output


class MegaCRNWrapper(nn.Module):
    """简化包装类，完全兼容你的训练接口"""

    def __init__(self, input_dim=7, output_dim=6, site=134, his=60):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.site = site
        self.his = his

        # 创建MegaCRN模型
        self.model = MegaCRNForPowerPred(
            num_nodes=site,
            input_dim=input_dim,  # 输入特征数
            output_dim=1,  # 每个时间步预测一个功率值
            horizon=output_dim,  # 预测步数
            rnn_units=64,
            num_layers=2,
            cheb_k=3,
            mem_num=20,
            mem_dim=64,
            use_curriculum_learning=True,
            cl_decay_steps=2000
        )

    def forward(self, x):
        """
        x: shape (batch_size, num_stations, seq_len, input_dim)
        输出: shape (batch_size, num_stations, output_dim)
        """
        return self.model(x)


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = MegaCRNWrapper(input_dim=7, output_dim=24, site=134, his=60)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 假设输入数据
    batch_size = 32
    x = torch.randn(batch_size, 134, 60, 7)  # [batch, sites, timesteps, features]

    # 前向传播
    output = model(x)  # 输出 [32, 134, 6]
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


