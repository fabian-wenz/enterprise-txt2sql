from transformers import AutoTokenizer, AutoModel, set_seed
import os
from tqdm import tqdm
import torch
from torch.utils import data
import numpy as np
import pickle
import json
import pandas as pd
from collections import Counter
from angle_emb import AnglE, Prompts
import torch.nn.functional as F
from typing import List
import sys

sys.path.append(os.path.abspath('./'))

#from src.scripts.utils import TextDataset, mean_pooling, create_directory, get_gold_t
from src.scripts.constants import *

# we can either represent tables as flattened schema or flattened schema + column descriptions
'''
def evaluate(data_dir, lm, test_scores, k, focus_q_idxs=None, verbose=False):
  with open('./queries.json', 'r') as f:
    qs = json.load(f)
  
  tables_df = pd.read_csv('./tables.csv')

  num_q = test_scores.shape[0]
  print(f'num_q: {num_q}')
  preds = np.argpartition(test_scores, -k, axis=1)[:,-k:]

  scores = np.array([0.0, 0.0, 0.0])
  
  pd.set_option('max_colwidth', 200)

  recall_list = []

  for q_idx in range(num_q):
    gold = set(get_gold_t(qs[q_idx]['sql']))
    pred = preds[q_idx]
    pred = tables_df.iloc[pred]['schema'].tolist()
    pred = set([x.split(',')[0] for x in pred])

    true_positives = len(gold.intersection(pred))
    false_positives = len(pred) - true_positives
    false_negatives = len(gold) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    
    scores += np.array([precision, recall, f1_score])
    recall_list.append(recall)

    if verbose:
    #if verbose and recall == 0:
      print(q_idx)
      print(qs[q_idx]['questions'])
      print(preds[q_idx])
      print(tables_df.iloc[preds[q_idx]])
      print(gold)
      #print(t_df.iloc[combs[0]])
      #print(_recall_list[best_idx])
      print(f'precision: {precision}, recall: {recall}, f1: {f1_score}')
      print('\n')
  
  scores = 100 * scores / num_q
  scores = np.round(scores, decimals=1)
  print(scores)
  print(Counter(recall_list))'''

# def embed_contriever(q, t, q_fn, t_fn, decomp):
#   device = 'cuda'
#   tokenizer = AutoTokenizer.from_pretrained('contriever-msmarco')
#   model = AutoModel.from_pretrained('contriever-msmarco').to(device)
#   print(f'max t length: {max([len(_t) for _t in t])}')

#   def collate_tokenize(data):
#     return tokenizer(data, padding=True, truncation=True, return_tensors='pt').to('cuda')

#   q_t_embeds = []
#   for texts in [q, t]:
#     test_dataset = TextDataset(texts)
#     test_dataloader = data.DataLoader(test_dataset, batch_size=100, shuffle=False, collate_fn=collate_tokenize)

#     text_embeds = []
#     model.eval()
#     with torch.no_grad():
#       for test_input in tqdm(test_dataloader):
#         output = model(**test_input)
#         embeds = mean_pooling(output[0], test_input['attention_mask'])
#         text_embeds.append(embeds)
#       text_embeds = torch.vstack(text_embeds)
#       q_t_embeds.append(text_embeds)
#       print(text_embeds.shape)

#   q_embeds, t_embeds = q_t_embeds
#   q_embeds = q_embeds.cpu()
#   t_embeds = t_embeds.cpu()

#   np.save(q_fn, q_embeds)
#   np.save(t_fn, t_embeds)
#   return q_embeds, t_embeds

def embed_uae(texts: List[str], fn: str, is_retrieval: bool):
  if os.path.isfile(fn):
    return torch.from_numpy(np.load(fn))

  angle = AnglE.from_pretrained('./UAE-Large-V1', pooling_strategy='cls').cuda()
  
  if is_retrieval:
    angle.set_prompt(prompt=Prompts.C)

  # if '-e' in lm:
  #   q[21] += ' Independent activities periods mean IAP.'
  #   q[28] += ' TDL means top_level_domain.'

  embeds = []
  for i in tqdm(range((len(texts)//BATCH_SIZE) + 1)):
    _texts = texts[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    
    if len(_texts) >= 1:
      if is_retrieval:
        _texts = [{'text': text} for text in _texts]
      vec = angle.encode(_texts, to_numpy=True)
      embeds.append(vec)
  
  embeds = np.vstack(embeds)

  np.save(fn, embeds)
  
  embeds = torch.from_numpy(embeds)

  return embeds

def get_sim_scores(q: List[str], t: List[str], dataset:str, is_retrieval: bool):
  save_filename = f'{PREFIX}/data/{dataset}/score.npy'

  q_embeds_fn = f'{PREFIX}/data/{dataset}/q_embeds.npy'
  t_embeds_fn = f'{PREFIX}/data/{dataset}/t_embeds.npy'

  print(save_filename)
  print(f'#q, #t: {len(q)}, {len(t)}')

  q_embeds = embed_uae(q, q_embeds_fn, is_retrieval)
  t_embeds = embed_uae(t, t_embeds_fn, is_retrieval)

  # get sim scores from embedding dot products
  if not os.path.isfile(save_filename):
    sim_scores = []
    for q_embed in q_embeds:
      sim_scores.append(F.cosine_similarity(q_embed.unsqueeze(0), t_embeds, dim=1).unsqueeze(0))
    sim_scores = torch.vstack(sim_scores).numpy()
    print(sim_scores.shape)
    
    np.save(save_filename, sim_scores)
  else:
    sim_scores = np.load(save_filename)

  return sim_scores

# k = -1 means for all tables
def top_k(dataset: str, q_idx: int, k: int) -> List[str]:
  save_filename = f'{PREFIX}/data/{dataset}/embeds/score.npy'
  if dataset == 'tig':
    save_filename = f'{PREFIX}/data/{dataset}/embeds/stella{DB_DBID}/score.npy'
  sim_scores = np.load(save_filename)

  if k == -1:
    top_k_indices = np.argsort(-sim_scores[q_idx])
  else:
    top_k_indices = np.argpartition(-sim_scores[q_idx], k)[:k]
    top_k_indices = top_k_indices[np.argsort(-sim_scores[q_idx][top_k_indices])]

  tables = pd.read_csv(f'{PREFIX}/data/{dataset}/tables.csv').iloc[top_k_indices]['schema'].tolist()
  tables = [x.split(',')[0] for x in tables]
  return tables

if __name__ == '__main__':
  set_seed(1234)

  dataset = 'mit'
  
  with open(f'{PREFIX}/data/{dataset}/queries.json', 'r') as f:
    questions = json.load(f)
  questions = [x['question'] for x in questions]
  t = pd.read_csv(f'{PREFIX}/data/{dataset}/tables.csv')['schema'].tolist()

  # get_sim_scores(questions, t, dataset, True)

  top_k(dataset, 0, -1)
