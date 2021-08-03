import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from KhaiPhaWeb.KhaiPhaWeb.AwesomeRecSystemModels.util import  train_model_util_PyTorch
from modelsummary import summary

AID_DATA_DIR = '../data/Criteo/forOtherModels/'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PNN_layer(nn.Module):
    def __init__(self, num_feat, num_field, dropout_deep, deep_layer_sizes, product_layer_dim=10,
                 reg_l1=0.01, reg_l2=1e-5, embedding_size=10, product_type='outer'):
        super().__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.num_feat = num_feat                                   # Denoted as
        self.num_field = num_field                                 # Denoted as N
        self.product_layer_dim = product_layer_dim                 # Denoted as D1
        self.dropout_deep = dropout_deep

        # Embedding (num_feat = len(feat)+1 sấp sỉ 73000,embeddingsize = 10)
        feat_embeddings = nn.Embedding(num_feat, embedding_size)
        nn.init.xavier_uniform_(feat_embeddings.weight)
        self.feat_embeddings = feat_embeddings

        # linear part
        linear_weights = torch.randn((product_layer_dim, num_field, embedding_size))   # D1 * N * M (num_field=39, embedding_size=10) 3900
        nn.init.xavier_uniform_(linear_weights)
        self.linear_weights = nn.Parameter(linear_weights)

        # quadratic part   product_layer_dim=10
        self.product_type = product_type
        if product_type == 'inner':
            theta = torch.randn((product_layer_dim, num_field))        # D1 * N (390)
            nn.init.xavier_uniform_(theta)
            self.theta = nn.Parameter(theta)
        else:
            quadratic_weights = torch.randn((product_layer_dim, embedding_size, embedding_size))  # D1 * M * M (1000)
            nn.init.xavier_uniform_(quadratic_weights)
            self.quadratic_weights = nn.Parameter(quadratic_weights)

        # fc layer
        self.deep_layer_sizes = deep_layer_sizes
        all_dims = [self.product_layer_dim + self.product_layer_dim] + deep_layer_sizes  #(20,400,400)

        for i in range(1, len(deep_layer_sizes) + 1):
            setattr(self, 'linear_' + str(i), nn.Linear(all_dims[i - 1], all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout_deep[i]))

        # last layer
        self.fc = nn.Linear(deep_layer_sizes[-1], 1)

    def forward(self, feat_index, feat_value, use_dropout=False):
        # embedding part
        feat_embedding = self.feat_embeddings(feat_index)          # Batch * N * M
        # linear part
        lz = torch.einsum('bnm,dnm->bd', feat_embedding, self.linear_weights)  # Batch * D1

        # quadratic part
        if self.product_type == 'inner':
            theta = torch.einsum('bnm,dn->bdnm', feat_embedding, self.theta)            # Batch * D1 * N * M
            lp = torch.einsum('bdnm,bdnm->bd', theta, theta)
        else:
            embed_sum = torch.sum(feat_embedding, dim=1)
            p = torch.einsum('bm,bn->bmn', embed_sum, embed_sum)
            lp = torch.einsum('bmn,dmn->bd', p, self.quadratic_weights)        # Batch * D1

        #Concat lz, lp Output = (2048,20) tức là Batch*(2*D1)
        y_deep = torch.cat((lz, lp), dim=1)

        if use_dropout:
            y_deep = nn.Dropout(self.dropout_deep[0])(y_deep)
        for i in range(1, len(self.deep_layer_sizes) + 1):
            y_deep = getattr(self, 'linear_' + str(i))(y_deep)
            y_deep = getattr(self, 'batchNorm_' + str(i))(y_deep)
            y_deep = torch.sigmoid(y_deep)
            if use_dropout:
                y_deep = getattr(self, 'dropout_' + str(i))(y_deep)
        output = self.fc(y_deep)
        return output

if __name__ == '__main__':
    train_data_path, test_data_path = AID_DATA_DIR + 'train_data/', AID_DATA_DIR + 'test_data/'
    feat_dict_ = pickle.load(open(AID_DATA_DIR + 'aid_data/feat_dict_10.pkl2', 'rb'))
    pnn = PNN_layer(num_feat=len(feat_dict_) + 1, num_field=39, dropout_deep=[0.5, 0.5, 0.5],
                    deep_layer_sizes=[400, 400], product_layer_dim=10,
                    reg_l1=0.01, reg_l2=1e-5, embedding_size=10, product_type='outer').to(DEVICE)
    summary(PNN_layer(num_feat=len(feat_dict_) + 1, num_field=39, dropout_deep=[0.5, 0.5, 0.5],
                    deep_layer_sizes=[400, 400], product_layer_dim=10,
                    reg_l1=0.01, reg_l2=1e-5, embedding_size=10, product_type='outer'),torch.zeros([2048, 39], dtype=torch.int32),torch.zeros([2048, 39], dtype=torch.int32), show_input=True,show_hierarchical=True)
    train_model_util_PyTorch.train_test_model_demo(pnn, DEVICE, train_data_path, test_data_path, feat_dict_)

