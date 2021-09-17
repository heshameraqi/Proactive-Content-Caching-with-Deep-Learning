import torch
import torch.nn as nn
import torch.nn.functional as F
from models.NCF_Att.attention_module import Transformer
#from ..NCF.NCF import NCF


class NCF_Att(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers,
                 dropout, model, GMF_model=None, MLP_model=None):
        super(NCF_Att, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors;
        num_layers: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
        GMF_model: pre-trained GMF weights;
        MLP_model: pre-trained MLP weights.
        """
        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(item_num, factor_num * (2 ** (num_layers - 1)))
        # Self Attention module:
        d_model = factor_num * (2 ** (num_layers - 1))
        self.user_att = Transformer(d_model=d_model, nhead=8, num_encoder_layers=3, dim_feedforward=d_model)
        self.item_att = Transformer(d_model=d_model, nhead=8, num_encoder_layers=3, dim_feedforward=d_model)

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        predict_size = factor_num
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        if self.model == 'NCF_Att-pre':
            # embedding layers
            self.embed_user_MLP.weight.data.copy_(
                self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(
                self.MLP_model.embed_item_MLP.weight)

            # mlp layers
            for (m1, m2) in zip(
                    self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            self.predict_layer.weight.data.copy_(self.MLP_model.predict_layer.weight)
            self.predict_layer.bias.data.copy_(self.MLP_model.predict_layer.bias)
        else:
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight,
                                     a=1, nonlinearity='sigmoid')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, user, item):
        embed_user_MLP = self.embed_user_MLP(user)
        embed_user_MLP = embed_user_MLP.unsqueeze(1)
        embed_user_MLP = self.user_att(embed_user_MLP)

        embed_item_MLP = self.embed_item_MLP(item)
        embed_item_MLP = embed_item_MLP.unsqueeze(1)
        embed_item_MLP = self.item_att(embed_item_MLP)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)

        concat = output_MLP

        prediction = self.predict_layer(concat)
        return prediction.view(-1)
