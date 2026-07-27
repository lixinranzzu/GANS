import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
try:
    torch.jit._state.disable()
except Exception:
    pass
import torch._jit_internal as _ji
if hasattr(torch.jit, "_overload"):
    torch.jit._overload = lambda f: f
if hasattr(_ji, "_check_overload_body"):
    _ji._check_overload_body = lambda f: None
# ---------------------------------------------------------
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import  degree
from torch_geometric.utils import remove_self_loops
import torch_geometric

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization

class GANS(GeneralRecommender):
    def __init__(self, config, dataset):
        super(GANS, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']  # not used
        dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']

        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.k = 40
        self.aggr_mode = 'add'
        self.dataset = dataset
        self.dropout = config['dropout']
        # self.construction = 'weighted_max'
        self.reg_weight = config['reg_weight']


        self.temp = config['temp']
        #self.beta = config['beta']
        self.v_rep = None
        self.t_rep = None
        self.v_preference = None
        self.t_preference = None
        self.id_preference = None
        self.dim_latent = 64
        self.dim_feat = 128
        self.mm_adj = None
        self.v1 = None
        self.t1 = None
        self.id1 = None
        self.cl_loss = config['cl_loss']
        self.alpha=config['alpha']
        self.beta=config['beta']

        self.mlp = nn.Linear(2*dim_x, 2*dim_x)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        #self.user_graph_dict = np.load(os.path.join(dataset_path, config['user_graph_dict_file']),
                                      #allow_pickle=True).item()

        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)
        self.mm_adj1 = self.build_personalized_ii_graph(
            user_seq_file=os.path.join(dataset_path, 'microlens_user_seq.csv'),
            num_items=self.num_item,
            device=self.device
        )
        self.mm_adj = (1-self.beta)*self.mm_adj+self.beta*self.mm_adj1

        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # pdb.set_trace()
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

        self.weight_i = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_item, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_i.data = F.softmax(self.weight_i, dim=1)

        self.item_index = torch.zeros([self.num_item], dtype=torch.long)




        self.MLP_user = nn.Linear(self.dim_latent * 2, self.dim_latent)

        if self.v_feat is not None:
            self.v_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.v_feat)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.t_feat)

        self.id_feat = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_items, self.dim_latent), dtype=torch.float32,
                                                requires_grad=True), gain=1).to(self.device))
        self.id_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                          dim_latent=64, device=self.device, features=self.id_feat)


        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)

    def build_personalized_ii_graph(self,user_seq_file, num_items, device='cpu'):
        """
        生成个性化 item-item 图
        user_seq_file: CSV, 每行 userID, item_seq
        num_items: 总物品数
        """
        df = pd.read_csv(user_seq_file)
        edges = []

        for seq in df['item_seq']:
            # 将字符串转换为列表
            item_list = eval(seq)
            # 遍历相邻 item
            for i in range(len(item_list) - 1):
                src, dst = item_list[i], item_list[i + 1]
                if src == 0 or dst == 0:
                    continue
                edges.append((src, dst))

        # 统计边出现次数
        edges = np.array(edges)
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)

        row = unique_edges[:, 0]
        col = unique_edges[:, 1]
        data = counts / counts.sum()  # 归一化为概率，可选

        # 构造稀疏矩阵
        ii_adj = sp.coo_matrix((data, (row, col)), shape=(num_items, num_items))

        # 转为 torch.sparse.FloatTensor
        indices = torch.tensor(np.vstack([ii_adj.row, ii_adj.col]), dtype=torch.long).to(device)
        values = torch.tensor(ii_adj.data, dtype=torch.float32).to(device)
        ii_adj_torch = torch.sparse.FloatTensor(indices, values, torch.Size([num_items, num_items]))

        return ii_adj_torch

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pre_epoch_processing(self):
        #self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        #self.user_weight_matrix = self.user_weight_matrix.to(self.device)
        pass


    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def hard_negative_bpr(self, user_emb, pos_emb, neg_emb):
        pos_expanded = pos_emb.unsqueeze(1).expand_as(neg_emb)
        mixed_neg_embeddings = self.alpha * pos_expanded + (1 - self.alpha) * neg_emb
    
        u_expanded = user_emb.unsqueeze(1).expand_as(mixed_neg_embeddings)
        scores = torch.sum(u_expanded * mixed_neg_embeddings, dim=2)  # [B, n_neg]
        hardest_indices = torch.argmax(scores, dim=1)
        hardest_neg_embeddings = mixed_neg_embeddings[torch.arange(len(user_emb)), hardest_indices]
        return hardest_neg_embeddings, hardest_indices


    def PMRL_LM_gram(self,view1, view2, view3, temperature=0.05):
        """
        三模态 PMRL alignment loss (使用 Gram 矩阵特征值，不分块版本)

        Args:
            view1, view2, view3: [B, D] 三个模态的嵌入
            temperature: τ

        Returns:
            loss: 标量
        """

        # Step 1：模态归一化
        v1 = F.normalize(view1, dim=1)
        v2 = F.normalize(view2, dim=1)
        v3 = F.normalize(view3, dim=1)

        # Step 2：构造 Z: [B, D, 3]
        Z = torch.stack([v1, v2, v3], dim=2)  # 最后一维是模态维

        # Step 3：Gram矩阵：G = Z^T Z    -> [B, 3, 3]
        G = torch.matmul(Z.transpose(1, 2), Z)

        # Step 4：求特征值（对称矩阵，因此用 eigvalsh 更快更稳定）
        # eigenvalues: [B, 3] (升序)
        eigvals = torch.linalg.eigvalsh(G)

        # Step 5：裁掉数值噪声（避免负值导致 sqrt 出 NaN）
        eigvals = torch.clamp(eigvals, min=0.0)

        # Step 6：转为降序（σ1 是最大的）
        eigvals_desc, _ = torch.sort(eigvals, dim=1, descending=True)

        # Step 7：奇异值 = sqrt(特征值)
        sv = torch.sqrt(eigvals_desc)  # [B, 3]

        # Step 8：CrossEntropy（论文使用此方式）
        logits = sv / temperature  # [B, 3]
        targets = torch.zeros(view1.size(0), dtype=torch.long, device=view1.device)

        loss = F.cross_entropy(logits, targets)

        return loss



    def forward(self, interaction):

        # -------------------------------------------------
        # 2. GCN encoding for each modality
        # -------------------------------------------------
        v_rep, _ = self.v_gcn(self.edge_index, self.v_feat)
        t_rep, _ = self.t_gcn(self.edge_index, self.t_feat)
        id_rep, _ = self.id_gcn(self.edge_index, self.id_feat)

        # store raw representations
        self.v1 = v_rep
        self.t1 = t_rep
        self.id1 = id_rep

        # -------------------------------------------------
        # 3. User multimodal fusion
        # -------------------------------------------------
        v_user = v_rep[:self.num_user]
        t_user = t_rep[:self.num_user]

        v_user = v_user.unsqueeze(2)
        t_user = t_user.unsqueeze(2)

        user_modal_stack = torch.cat((v_user, t_user), dim=2)
        weighted_user_modal = self.weight_u.transpose(1, 2) * user_modal_stack

        user_rep = torch.cat(
            (weighted_user_modal[:, :, 0], weighted_user_modal[:, :, 1]),
            dim=1
        )


        # -------------------------------------------------
        # 4. Item representation + item-item graph enhancement
        # -------------------------------------------------
        v_item = v_rep[self.num_user:]
        t_item = t_rep[self.num_user:]

        v_item_graph = self.buildItemGraph(v_item)
        t_item_graph = self.buildItemGraph(t_item)

        v_item = v_item + v_item_graph
        t_item = t_item + t_item_graph

        item_rep = torch.cat((v_item, t_item), dim=1)


        # 5. Final embeddings
        # -------------------------------------------------
        self.user_rep = user_rep
        self.item_rep = item_rep
        #self.user_rep = v_user
        #self.item_rep = v_item

        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        result_embed_v = torch.cat((self.v1[:self.num_user], v_item), dim=0)
        result_embed_t = torch.cat((self.t1[:self.num_user], t_item), dim=0)

        return result_embed_v,result_embed_t

    def buildItemGraph(self, h):
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h


    def calculate_loss(self, interaction):
        user_idx, pos_item_idx, neg_item_idx = interaction
        pos_item_idx = pos_item_idx + self.n_users
        neg_item_idx = neg_item_idx + self.n_users
        result_embed_v,result_embed_t = self.forward(interaction)
        # -------------------------------------------------
        #  Fetch embeddings for training triples
        # -------------------------------------------------
        user_embed = self.result_embed[user_idx]

        user_embed_v = result_embed_v[user_idx]
        user_embed_t = result_embed_t[user_idx]

        pos_item_embed = self.result_embed[pos_item_idx]

        pos_item_embed_v = result_embed_v[pos_item_idx]
        pos_item_embed_t = result_embed_t[pos_item_idx]

        neg_item_embed_v = result_embed_v[neg_item_idx]
        neg_item_embed_t = result_embed_t[neg_item_idx]
        # -------------------------------------------------
        #  Hard negative mining
        # -------------------------------------------------
        neg_item_v, _ = self.hard_negative_bpr(
            user_embed_v, pos_item_embed_v, neg_item_embed_v
        )

        neg_item_t, _ = self.hard_negative_bpr(
            user_embed_t, pos_item_embed_t, neg_item_embed_t
        )

        neg_item_embed = torch.cat((neg_item_v, neg_item_t), dim=1)
        #neg_item_embed=neg_item_v

        # -------------------------------------------------
        #  BPR scores
        # -------------------------------------------------
        pos_scores = torch.sum(user_embed * pos_item_embed, dim=1)
        neg_scores = torch.sum(user_embed * neg_item_embed, dim=1)
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        # -------------------------------------------------
        #  Multimodal contrastive loss
        # -------------------------------------------------
        cl_loss = (
                self.PMRL_LM_gram(self.id1[user_idx], self.v1[user_idx], self.t1[user_idx])
                + self.PMRL_LM_gram(
            self.id1[pos_item_idx],
            self.v1[pos_item_idx],
            self.t1[pos_item_idx],
        )
        )

        # reg
        reg_embedding_loss_v = (self.v_preference[user_idx] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[user_idx] ** 2).mean() if self.t_preference is not None else 0.0
        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
        reg_loss += self.reg_weight * (self.weight_u ** 2).mean()


        return loss_value + reg_loss+self.cl_loss*cl_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix



class GCN(torch.nn.Module):
    def __init__(self, datasets, batch_size, num_user, num_item, dim_id, aggr_mode,
                 dim_latent=None, device=None, features=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.device = device

        if self.dim_latent:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
            self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

        else:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_feat), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index, features, perturbed=False):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)

        h = self.conv_embed_1(x, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h).cuda()
            h += torch.sign(h) * F.normalize(random_noise, dim=-1) * 0.1
        h_1 = self.conv_embed_1(h, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h).cuda()
            h_1 += torch.sign(h_1) * F.normalize(random_noise, dim=-1) * 0.1
        # h_2 = self.conv_embed_1(h_1, edge_index)

        x_hat = x + h + h_1
        return x_hat, self.preference


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        # pdb.set_trace()
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # pdb.set_trace()
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            # pdb.set_trace()
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
