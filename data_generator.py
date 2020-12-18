import pickle
import os
import random
from transformers import BertModel, BertTokenizer
import numpy as np
import torch
SEED=0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
from tqdm import tqdm
dtype=torch.long
# info_dict = {'bug_severity': 7, 'bug_status': 3, 'component': 119, 'priority': 5, 'product': 54, 'version': 43}
info_dict = {'bug_severity': 6, 'bug_status': 3, 'component': 89, 'priority': 5, 'product':36 , 'version': 490}

# info_dict = {'bug_severity': 7, 'bug_status': 3, 'component': 612, 'priority': 5, 'product': 170, 'version': 326}
from keras.preprocessing.sequence import pad_sequences

print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def to_one_hot(idx, size):
  one_hot = np.zeros(size)
  one_hot[int(float(idx))] = 1
  return one_hot

def read_bug_ids(data):
  bug_ids = []
  with open(os.path.join(data, 'bug_ids.txt'), 'r') as f:
    for line in f:
      bug_ids.append(int(line.strip()))
  return bug_ids

def get_neg_bug(invalid_bugs, bug_ids):
  neg_bug = random.choice(bug_ids)
  while neg_bug in invalid_bugs:
    neg_bug = random.choice(bug_ids)
  return neg_bug



def read_batch_bugs(batch_bugs, data,test=True):
  desc_word = []
  short_desc_word = []
  info = []
  for bug_id in batch_bugs:
    bug = pickle.load(open(os.path.join('/content/drive/My Drive/DuplicateBugFinder/openOffice/bugs',  '{}.pkl'.format(bug_id)), 'rb'))
    desc_word.append(bug['description_long'][0:500])
    short_desc_word.append(bug['description_short'])   
    info_ = np.concatenate((
      to_one_hot(bug['bug_severity'], info_dict['bug_severity']),
      to_one_hot(bug['bug_status'], info_dict['bug_status']),
      to_one_hot(bug['component'], info_dict['component']),
      to_one_hot(bug['priority'], info_dict['priority']),
      to_one_hot(bug['product'], info_dict['product']),
      to_one_hot(bug['version'], info_dict['version'])))
    info.append(info_)

  encoded_long = []
  encoded_short = []
  for input_long in desc_word:
     encoded_input_long = tokenizer.encode(input_long,add_special_tokens = True,)
     encoded_long.append(encoded_input_long)

  input_ids_long = pad_sequences(encoded_long, maxlen=500, dtype="long",value=0, truncating="post", padding="post")
  attention_masks_long = []

  for sent in input_ids_long :
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks_long.append(att_mask)
  attention_masks_long = torch.tensor( attention_masks_long).cuda()
  encoded_long = torch.tensor( input_ids_long ).cuda()

  for input_short in short_desc_word:
     encoded_input_short = tokenizer.encode(input_short,add_special_tokens = True,)
     encoded_short.append(encoded_input_short)
  input_ids_short = pad_sequences(encoded_short, maxlen=30, dtype="long",value=0, truncating="post", padding="post")
  attention_masks_short = []
  for sent in input_ids_short :
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks_short.append(att_mask)    
  attention_masks_short = torch.tensor(attention_masks_short).cuda()
  encoded_short = torch.tensor(input_ids_short).cuda()
  info = torch.from_numpy(np.array(info)).type(dtype)cuda()
  batch_bugs = dict()
  batch_bugs['info'] = info
  batch_bugs['desc'] = (encoded_long,attention_masks_long)
  batch_bugs['short_desc'] = (encoded_short,attention_masks_short)
 

 
  

  return batch_bugs
 

def read_batch_triplets(batch_triplets, data):
  batch_input_bugs = []
  batch_pos_bugs = []
  batch_neg_bugs = []

  for triplet in batch_triplets:
    batch_input_bugs.append(triplet[0])
    batch_pos_bugs.append(triplet[1])
    batch_neg_bugs.append(triplet[2])

   
  return read_batch_bugs(batch_input_bugs, data), \
         read_batch_bugs(batch_pos_bugs, data), \
         read_batch_bugs(batch_neg_bugs, data)


def read_test_data(data):
  test_data = []
  bug_ids = set()
  with open(os.path.join(data, 'test.txt'), 'r') as f:
    for line in f:
      tokens = line.strip().split()
      test_data.append([int(tokens[0]), [int(bug) for bug in tokens[1:]]])
      for token in tokens:
        bug_ids.add(int(token))
  return test_data, list(bug_ids)


def read_train_data(data):
  data_pairs = []
  data_dup_sets = {}
  with open(os.path.join(data, 'train.txt'), 'r') as f:
    for line in f:
      bug1, bug2 = line.strip().split()
      data_pairs.append([int(bug1), int(bug2)])
      if int(bug1) not in data_dup_sets.keys():
        data_dup_sets[int(bug1)] = set()
      data_dup_sets[int(bug1)].add(int(bug2))
  return data_pairs, data_dup_sets
#找出相似的set 和 配对 

train_data = None
dup_sets = None


def batch_iterator(data, batch_size, n_neg):
  global train_data
  global dup_sets
  if not train_data:
    train_data, dup_sets = read_train_data(data)#用train set 去 训练数据 得到的是 ID
  random.shuffle(train_data)#重新排序 
  bug_ids = read_bug_ids(data)#得到bugID
  num_batches = int(len(train_data) / batch_size)#循环几次
  if len(data) % batch_size > 0:
    num_batches += 1
  loop = tqdm(range(num_batches))
  loop.set_description('Training')
  for i in range (num_batches):
    batch_triplets = []
    for k in range(batch_size):
      next_training = i *batch_size + k
      if next_training >= len(train_data):
         break
      for i in range(n_neg):
        neg_bug = get_neg_bug(dup_sets[train_data[next_training][0]], bug_ids)#返回一个不醒一样的 bug
        batch_triplets.append([train_data[next_training][0], train_data[next_training][1], neg_bug])
    yield loop, read_batch_triplets(batch_triplets, data)


