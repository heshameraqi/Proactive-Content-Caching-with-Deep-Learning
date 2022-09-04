import torch
import torch.nn as nn
import torch.nn.functional as F 
import config


class NCF_with_U_I_info(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout,
     model, GMF_model=None, MLP_model=None, user_info_num=0, item_info_num=0):
        super(NCF_with_U_I_info, self).__init__()
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

        MLP_modules = []
        for i in range(num_layers):
          input_size = factor_num * (2 ** (num_layers - i))
          MLP_modules.append(nn.Dropout(p=self.dropout))
          MLP_modules.append(nn.Linear(input_size, input_size//2))
          MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if config.user_item_info:
          self.item_info_embed = nn.Sequential(
                                        nn.Linear(item_info_num, factor_num),
                                        nn.ReLU()
                                        )
          self.user_info_embed = nn.Sequential(
                                        nn.Linear(user_info_num, factor_num),
                                        nn.ReLU()
                                        )

        if config.user_item_info:
          if self.model in ['MLP', 'GMF']:
            predict_size = factor_num * 3
          else:
            predict_size = factor_num * 4
        else:
          if self.model in ['MLP', 'GMF']:
            predict_size = factor_num
          else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        if config.pretrain:
              if config.model == 'GMF' or config.model == 'NeuMF-pre':
                  # embedding layers
                  self.embed_user_GMF.weight.data.copy_(
                          self.GMF_model.embed_user_GMF.weight)
                  self.embed_item_GMF.weight.data.copy_(
                          self.GMF_model.embed_item_GMF.weight)
              if config.model == 'MLP' or config.model == 'NeuMF-pre':
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

              if config.model == 'NeuMF-pre':
                  for (m1, m2, m3) in zip(
                          self.item_info_embed, self.GMF_model.item_info_embed, self.MLP_model.item_info_embed):
                      if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear) and isinstance(m3, nn.Linear):
                          item_info_embed_weight = m2.weight + m3.weight
                          item_info_embed_bias = m2.bias + m3.bias
                          m1.weight.data.copy_(0.5 * item_info_embed_weight)
                          m1.bias.data.copy_(0.5 * item_info_embed_bias)

                  for (m1, m2, m3) in zip(
                          self.user_info_embed, self.GMF_model.user_info_embed, self.MLP_model.user_info_embed):
                      if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear) and isinstance(m3, nn.Linear):
                          user_info_embed_weight = m2.weight + m3.weight
                          user_info_embed_bias = m2.bias + m3.bias
                          m1.weight.data.copy_(0.5 * user_info_embed_weight)
                          m1.bias.data.copy_(0.5 * user_info_embed_bias)

                  # predict layers
                  predict_weight_1 = torch.cat([
                      self.GMF_model.predict_layer.weight[:, :32],
                      self.MLP_model.predict_layer.weight[:, :32]], dim=1)
                  predict_weight_2 = self.GMF_model.predict_layer.weight[:, 32:] +\
                                     self.MLP_model.predict_layer.weight[:, 32:]
                  predict_weight = torch.cat([predict_weight_1, 0.5*predict_weight_2], dim=1)
                  precit_bias = self.GMF_model.predict_layer.bias + \
                                self.MLP_model.predict_layer.bias

                  self.predict_layer.weight.data.copy_(predict_weight)
                  self.predict_layer.bias.data.copy_(0.5 * precit_bias)
              else:
                  for m in self.item_info_embed:
                        if isinstance(m, nn.Linear):
                          nn.init.xavier_uniform_(m.weight)

                  for m in self.user_info_embed:
                    if isinstance(m, nn.Linear):
                      nn.init.xavier_uniform_(m.weight)
              
                  # predict layers
                  nn.init.kaiming_uniform_(self.predict_layer.weight,
                                  a=1, nonlinearity='sigmoid')
        else:
              nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
              nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
              nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
              nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

              for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                  nn.init.xavier_uniform_(m.weight)

              for m in self.item_info_embed:
                if isinstance(m, nn.Linear):
                  nn.init.xavier_uniform_(m.weight)
              
              for m in self.user_info_embed:
                if isinstance(m, nn.Linear):
                  nn.init.xavier_uniform_(m.weight)
              
              nn.init.kaiming_uniform_(self.predict_layer.weight, 
                          a=1, nonlinearity='sigmoid')

              for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                  m.bias.data.zero_()
          

    def forward(self, user, item, user_info, item_info):
        if not self.model == 'MLP':
          embed_user_GMF = self.embed_user_GMF(user)
          embed_item_GMF = self.embed_item_GMF(item)
          output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
          embed_user_MLP = self.embed_user_MLP(user)
          embed_item_MLP = self.embed_item_MLP(item)
          interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
          output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
          concat = output_GMF
        elif self.model == 'MLP':
          concat = output_MLP
        else:
          concat = torch.cat((output_GMF, output_MLP), -1)

        if config.user_item_info:
          item_info_embed = self.item_info_embed(item_info)
          user_info_embed = self.user_info_embed(user_info)
          concat2 = torch.cat((item_info_embed, user_info_embed), -1)
          concat = torch.cat((concat, concat2), -1)                    

        prediction = self.predict_layer(concat)
        return prediction.view(-1)
