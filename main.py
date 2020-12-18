import argparse
import sys
from google.colab import files
import torch.optim as optim
from torch import nn 

import data_generator
import proposed
from data_generator import *
import torch.nn.functional as F

SEED=0
import torch.nn as nn
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("GPU")
else:
  device = torch.device("cpu")
  print("CPU")
import matplotlib.pyplot as plt
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='/content/drive/My Drive/DuplicateBugFinder/openOffice/')
parser.add_argument('-k', '--top_k', type=int, default=25)
parser.add_argument('-e', '--epochs', type=int, default=30)

parser.add_argument('-nf', '--n_filters', type=int, default=300)
parser.add_argument('-np', '--n_prop', type=int, default=629)#1123#629#231#452
parser.add_argument('-bs', '--batch_size', type=int, default=8)#
parser.add_argument('-nn', '--n_neg', type=int, default=1)
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5, metavar='LR',
                    help='learning rate (default: 1e-3)') #5e-5
args = parser.parse_args()

def export(net, data):
  _, bug_ids = read_test_data(data)
  features = {}
  batch_size = args.batch_size * args.n_neg
  num_batch = int(len(bug_ids) / batch_size)
  if len(bug_ids) % batch_size > 0:
    num_batch += 1
  loop = tqdm(range(num_batch))
 
  with torch.no_grad():
    for i in loop:
      batch_ids = []
      for j in range(batch_size):
        offset = batch_size * i + j
        if offset >= len(bug_ids):
          break
        batch_ids.append(bug_ids[offset])
      batch_features = net(read_batch_bugs(batch_ids, data, test=True))
      for bug_id, feature in zip(batch_ids, batch_features):
        features[bug_id] = feature
  return features


def test(data, top_k, features=None):
  if not features:
    features = torch.load(str(args.net) + '_features.t7')
  cosine_batch = nn.CosineSimilarity(dim=1, eps=1e-6)
  corrects = 0
  total = 0
  samples_ = torch.stack(tuple(features.values()))
  print('abc',samples_.shape)
  test_data, _ = read_test_data(data)
  loop = tqdm(range(len(test_data)))
  loop.set_description('Testing')
  for i in loop:
    idx = test_data[i]
    query = idx[0]
    ground_truth = idx[1]

    query_ = features[query].expand(samples_.size(0), samples_.size(1))
    cos_ = cosine_batch(query_, samples_)

    (_, indices) = torch.topk(cos_, k=top_k + 1)
    candidates = [list(features.keys())[x.item()] for x in indices]

    corrects += len(set(candidates) & set(ground_truth))
    total += len(ground_truth)


  return float(corrects) / total

def init_network(model, method='xavier', exclude='word_embed', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def main():

  args.net = 'proposed'

  if os.path.exists(str(args.net) + '_features.t7'):
    print('Final recall@{}={:.4f}'.format(args.top_k, test(args.data, args.top_k)))
    sys.exit(0)

  if os.path.exists(str(args.net) + '_checkpoint.t7'):
    net = torch.load(str(args.net) + '_checkpoint.t7')
    features = export(net, args.data)
    torch.save(features, str(args.net) + '_features.t7')
    print('Final recall@{}={:.4f}'.format(args.top_k, test(args.data, args.top_k)))
    sys.exit(0)
 

  net = proposed.Net(args)
  net = nn.DataParallel(net)
  net.cuda()

  # init_network(net)
  
  best_recall = 0
  best_epoch = 0
  losse =[]
  acc = []
  net.train()
  margin = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))

  optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
  for epoch in range(1, args.epochs + 1):
      result= 0
      print('Epoch: {}'.format(epoch))
      for loop, (batch_x, batch_pos, batch_neg) in data_generator.batch_iterator(args.data, args.batch_size, args.n_neg):
          losses =[]
          net.zero_grad()
          pred = net(batch_x)
          pred_pos = net(batch_pos)
          pred_neg = net(batch_neg)
          
          loss = margin(pred, pred_pos, pred_neg)
          losses.append(loss.item())
          loop.set_postfix(loss=loss.item())
        
          loss.backward()
          optimizer.step()
          result = np.array(losses).mean()
    # if epoch == 10:
    #   optimizer = optim.Adam(net.parameters(), lr=args.learning_rate * 0.1)

     
      net.eval()
      features = export(net, args.data)
      recall = test(args.data, args.top_k, features)
      net.train()
      losse.append(result)
      acc.append(recall)
      print('Loss={:.4f}, Recall@{}={:.4f}'.format(loss, args.top_k, recall))
      if recall > best_recall:
        best_recall = recall
        best_epoch = epoch
        torch.save(net, str(args.net) + '_checkpoint.t7')
        torch.save(features, str(args.net) + '_features.t7')
  print('Best_epoch={}, Best_recall={:.4f}'.format(best_epoch, best_recall))
  plt.plot(losse)
  plt.plot(acc)
  images_dir = '/content/drive/My Drive/DuplicateBugFinder'
  plt.savefig(f"{images_dir}/abc.png")
if __name__ == "__main__":
  main()
