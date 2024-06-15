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

from src.scripts.utils import TextDataset, mean_pooling, create_directory, get_gold_t
from src.scripts.constants import *
from src.scripts.eval import Evaluator


# we can either represent tables as flattened schema or flattened schema + column descriptions

def evaluate(data_dir, lm, test_scores, k, focus_q_idxs=None, verbose=False):
  with open(f'{prefix}/data/{db_id}/queries.json', 'r') as f:
    qs = json.load(f)

  tables_df = pd.read_csv(f'{prefix}/data/{db_id}/tables.csv')

  num_q = test_scores.shape[0]
  print(f'num_q: {num_q}')
  preds = np.argpartition(test_scores, -k, axis=1)[:, -k:]


  pd.set_option('max_colwidth', 200)

  recall_list = []

  evaluator = Evaluator(len(qs), dict())
  golds = []
  predicts = []
  for q_idx in range(num_q):
    golds.append(set(get_gold_t(qs[q_idx]['sql'])))
    pred = preds[q_idx]
    pred = tables_df.iloc[pred]['schema'].tolist()
    predicts.append(set([x.split(',')[0] for x in pred]))
  evaluator.tables_gold_sqls = golds
  evaluator.tables_pred_sqls = predicts
  evaluator.generate_metrics(EvaluationType.TABLE_NAME_EXTRACTION)
  scores = evaluator.get_metric_summary(EvaluationType.TABLE_NAME_EXTRACTION)

  scores = 100 * np.array(scores)
  scores = np.round(scores, decimals=1)
  print(scores)
  print(Counter(recall_list))


def embed_uae(texts: List[str], fn: str, is_retrieval: bool):
  if os.path.isfile(fn):
    return torch.from_numpy(np.load(fn))

  angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

  #if is_retrieval:
  #  angle.set_prompt(prompt=Prompts.C)

  # if '-e' in lm:
  #   q[21] += ' Independent activities periods mean IAP.'
  #   q[28] += ' TDL means top_level_domain.'

  embeds = []
  for i in tqdm(range((len(texts) // BATCH_SIZE) + 1)):
    _texts = texts[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

    if len(_texts) >= 1:
      if is_retrieval:
        _texts = [{'text': text} for text in _texts]
        vec = angle.encode(_texts, to_numpy=True, prompt=Prompts.C)
      else:
        vec = angle.encode(_texts, to_numpy=True)
      embeds.append(vec)

  embeds = np.vstack(embeds)

  np.save(fn, embeds)

  embeds = torch.from_numpy(embeds)

  return embeds


def get_sim_scores(q: List[str], t: List[str], dataset: str, is_retrieval: bool):
  save_filename = f'{prefix}/data/{dataset}/score.npy'

  q_embeds_fn = f'{prefix}/data/{dataset}/q_embeds.npy'
  t_embeds_fn = f'{prefix}/data/{dataset}/t_embeds.npy'

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


def top_k(dataset: str, q_idx: int, k: int) -> List[str]:
  save_filename = f'{prefix}/data/{dataset}/embeds/score.npy'
  sim_scores = np.load(save_filename)

  top_k_indices = np.argpartition(-sim_scores[q_idx], k)[:k]
  top_k_indices = top_k_indices[np.argsort(-sim_scores[q_idx][top_k_indices])]

  tables = pd.read_csv(f'{prefix}/data/{dataset}/tables.csv').iloc[top_k_indices]['schema'].tolist()
  tables = [x.split(',')[0] for x in tables]
  return tables


if __name__ == '__main__':
  set_seed(1234)

  dataset = 'fiben'

  with open(f'{prefix}/data/{dataset}/queries.json', 'r') as f:
    questions = json.load(f)
  questions = [x['question'] for x in questions]
  t = pd.read_csv(f'{prefix}/data/{dataset}/tables.csv')['schema'].tolist()

  get_sim_scores(questions, t, dataset, True)

  top_k(dataset, 0, 5)
