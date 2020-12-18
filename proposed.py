import torch.nn as nn
from transformers import BertModel

from data_generator import *
import torch
import math
import torch.nn.functional as F

SEED=0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
class Residual(nn.Module):
  def __init__(self, d, fn):
    super(Residual, self).__init__()
    self.fn = fn
    self.projection = nn.Sequential(nn.Linear(d, d), fn, nn.Linear(d, d))

  def forward(self, x):
    return self.fn(x + self.projection(x))


class Net(nn.Module):
  def __init__(self, args):
    super(Net, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    for param in self.bert.parameters():
      param.requires_grad = False

    self.info_proj = nn.Sequential(nn.Linear(args.n_prop, 100), nn.Tanh())
    self.residual = Residual(300, nn.Tanh())
    self.projection = nn.Linear(300, 100)
 
    self.bert_proj = nn.Sequential(nn.Linear(768, 100), nn.Tanh())
  def forward_cnn(self, x):
    context = x[0]  
    mask = x[1]  
    pooled = self.bert(context, attention_mask=mask)
    return pooled.last_hidden_state 

  def forward_rnn(self, x):
    context = x[0]  
    mask = x[1]  
    pooled = self.bert(context, attention_mask=mask)
    return pooled.last_hidden_state 


  def forward(self, x):
    info = x['info']
    info_feature = self.info_proj(info.float())
    bert_long= self.forward_cnn(x['desc'])
    bert_long = torch.mean( bert_long, dim=1)
    bert_long = self.bert_proj(bert_long)
    bert_short = self.forward_rnn(x['short_desc'])
    bert_short = torch.mean( bert_short, dim=1)
    bert_short= self.bert_proj(bert_short)
  
    feature = torch.cat([info_feature,bert_long, bert_short], -1)
    # feature_res = self.residual(feature)
    return self.projection(feature)





