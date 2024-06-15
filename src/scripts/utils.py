import os
import random
import numpy as np
import torch
from torch.utils import data
import json
import pandas as pd
from numpy.linalg import norm
import re
from sql_metadata import Parser


def create_directory(path):
  if not os.path.exists(path):
    os.makedirs(path)


class TextDataset(data.Dataset):
  def __init__(self, texts):
    self.texts = texts

  def __getitem__(self, idx):
    return self.texts[idx]

  def __len__(self):
    return len(self.texts)


def mean_pooling(token_embeddings, mask):
  token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
  sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
  return sentence_embeddings


def cosine_sim(a, b):
  _a, _b = a.detach().cpu().numpy(), b.detach().cpu().numpy()
  return (_a @ _b) / (norm(_a) * norm(_b))


def get_gold_t(sql: str):
  gold_ts = Parser(sql).tables
  gold_ts = [gold_t.upper() for gold_t in gold_ts]
  for i in range(len(gold_ts)):
    if '.' in gold_ts[i]:
      gold_ts[i] = gold_ts[i].split('.')[1]

  return gold_ts
