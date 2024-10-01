import os
import sys

sys.path.append(os.path.abspath('./'))

import random
import numpy as np
import torch
from torch.utils import data
import json
import pandas as pd
from numpy.linalg import norm
import re
from sqlglot import parse_one, exp, transpile
from sql_metadata import Parser
from typing import Union, List

from src.scripts.constants import PREFIX

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
  return (_a @ _b)/(norm(_a) * norm(_b))

def sql_to_tables(dataset: str, sql: str) -> List[str]:
  gold_ts = Parser(sql).tables
  gold_ts = [gold_t.upper() for gold_t in gold_ts]
  for i in range(len(gold_ts)):
    if '.' in gold_ts[i]:
      gold_ts[i] = gold_ts[i].split('.')[1]
  # return gold_ts

  tables = os.listdir(f'{PREFIX}/data/{dataset}/schema')
  tables = set(table.replace('.csv', '') for table in tables if table.endswith('.csv'))

  if dataset == 'spider' and 'SINGER' in gold_ts:
    gold_ts_new = set(gold_ts) & tables
    gold_ts_new.add('SINGER')
  else:
    gold_ts_new = set(gold_ts) & tables
  # if set(gold_ts_new) != set(gold_ts):
  #   print(gold_ts)
  #   print(gold_ts_new)
  gold_ts = list(sorted(list(gold_ts_new)))
  return gold_ts


  # sql = sql.replace('`', '"')
  # transpile(sql)
  # parsed_query = parse_one(sql)
  # gold_ts = set(str(table) for table in parsed_query.find_all(exp.Table))
  # gold_ts = [gold_t.upper() for gold_t in gold_ts]
  # for i in range(len(gold_ts)):
  #   if 'AS' in gold_ts[i]:
  #     gold_ts[i] = gold_ts[i].split(' AS ')[0].strip()
  #   gold_ts[i] = gold_ts[i].replace('"', '')

  #   if '.' in gold_ts[i]:
  #     gold_ts[i] = gold_ts[i].split('.')[1]

  return gold_ts

class Execute():
  def __init__(self, model_name: str, cot: bool):
    from transformers import AutoTokenizer, set_seed, AutoModelForCausalLM, BitsAndBytesConfig
    from openai import OpenAI

    quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type='nf4',
      bnb_4bit_compute_dtype='float16',
    )

    set_seed(0)

    self.model_name = model_name
    self.cot = cot

    if model_name.startswith('gpt'):
      from src.scripts.constants import CONFIG

      self.client = OpenAI(
        api_key=CONFIG['api_key']
      )
    elif model_name.startswith('mistral'):
      tokenizer = AutoTokenizer.from_pretrained(f'mistralai/Mixtral-8x7B-Instruct-v0.1')
      tokenizer.pad_token = tokenizer.eos_token
      model = AutoModelForCausalLM.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1', device_map='auto', quantization_config=quantization_config)
      # model = AutoModelForCausalLM.from_pretrained(f'mistralai/Mixtral-8x7B-v0.1', device_map='auto', torch_dtype=torch.bfloat16)
      self.model = {'model': model, 'tokenizer': tokenizer}
    elif model_name.startswith('gemma'):
      print('bfloat16')
      model = AutoModelForCausalLM.from_pretrained('google/gemma-1.1-7b-it', device_map='auto', torch_dtype=torch.bfloat16)
      tokenizer = AutoTokenizer.from_pretrained('google/gemma-1.1-7b-it')
      self.model = {'model': model, 'tokenizer': tokenizer}
    elif model_name.startswith('llama'):
      model_id = 'meta-llama/Meta-Llama-3-8B-Instruct' if model_name == 'llama8' else 'meta-llama/Meta-Llama-3-70B-Instruct'
      if model_name == 'llama8':
        print('bfloat16')
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)
      else:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', quantization_config=quantization_config)
      tokenizer = AutoTokenizer.from_pretrained(model_id)
      self.model = {'model': model, 'tokenizer': tokenizer}

  # (key, p)
  def inference(self, r_idx:str=None, p:Union[str, dict]=None, prev_r:str=None, split_prompt=False, openai_batch=False):
    chat = [{'role': 'user', 'content': p}]

    if self.model_name.startswith('gpt'):
      messages=[{'role': 'user', 'content': p}]

      if openai_batch:
        return messages

      if prev_r:
        messages.append({'role': 'assistant', 'content': prev_r})

      resp = self.client.chat.completions.create(
        model=self.model_name,
        temperature=0,
        max_tokens=512,
        messages=messages
      )

      outputs = resp.choices[0].message.content

      if prev_r:
        outputs = prev_r + outputs

    if self.model_name.startswith('mistral'):
      tokenizer = self.model['tokenizer']
      model = self.model['model']

      inputs = tokenizer.apply_chat_template(chat, tokenize=False)
      #print(inputs)
      # default 256 is good except some which do not work, so re-run with 512 for those which do not work
      inputs = tokenizer(inputs, return_tensors='pt').to(model.device)
      outputs = model.generate(**inputs, max_new_tokens=1024 if self.cot else 512, pad_token_id=tokenizer.eos_token_id)
      outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
      outputs = outputs[0].split('[/INST]')[-1]
      # print(outputs)
      # print('-'*30)

    # prompt need to go into one chunk, can't divide them up
    if self.model_name.startswith('gemma'):
      tokenizer = self.model['tokenizer']
      model = self.model['model']

      prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
      # print(prompt)
      inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
      outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=1024 if self.cot else 256)
      outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
      outputs = outputs[0].split('model')[-1]
      # print(outputs)
      # print('-'*30)

    if self.model_name.startswith('llama'):
      tokenizer = self.model['tokenizer']
      model = self.model['model']

      prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
      # print(prompt)
      inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
      terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]
      outputs = model.generate(**inputs, max_new_tokens=1024 if self.cot else 512, eos_token_id=terminators, pad_token_id=tokenizer.eos_token_id)
      outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
      outputs = outputs[0].split('assistant')[-1]
      # print(outputs)
      # print('-'*30)

    return {r_idx: outputs} if r_idx is not None else outputs

def get_r_fn(dataset, model, k, primary, correct_tables, desc, num_example):
  key_active_string, correct_tables_str, desc_str = '', '', ''
  if primary:
    key_active_str = '_PRIM_KEYS'
  if correct_tables:
    correct_tables_str = '_CORRECT_TABLES'
  if desc:
    desc_str = '_DESC'
  result_fn = f'{PREFIX}/data/{dataset}/predictions/{model}/top{k}_examples_{num_example}{key_active_str}{correct_tables_str}{desc_str}.json'
  return result_fn

def sig_fig(x):
  return float(f'{float(f"{x:.3g}"):g}')

# long-running + zero-result
SKIP_IDXS = {
  'bird': [271, 330, 363, 405, 420, 455, 551, 586, 602, 613, 624, 637, 644, 671, 677, 774, 894, 943, 1006, 1009, 1098, 1112, 1114, 1121, 1122, 1127, 1138, 1466, 1480],
  'fiben': [202, 268, 269, 270, 278, 84, 128, 138, 146, 158, 175, 179, 186, 214, 215, 216, 222, 226, 232, 233, 234, 235, 236, 238, 239, 240, 241, 242, 243, 245, 252, 253, 254, 260, 263, 265, 266, 273, 275, 284, 285, 286] + [2, 3, 6, 10, 12, 18, 28, 30, 31, 33, 34, 35, 36, 39, 41, 43, 44, 47, 48, 51, 54, 55, 58, 62, 63, 65, 69, 70, 76, 78, 81, 82, 83, 87, 94, 95, 96, 97, 98, 101, 103, 104, 110, 112, 117, 118, 124, 125, 132, 135, 136, 147, 159, 160, 161, 162, 163, 164, 165, 166, 167, 172, 173, 174, 176, 177, 178, 180, 181, 200, 201, 203, 206, 210, 211, 212, 213, 217, 218, 223, 224, 225, 227, 228, 229, 230, 231, 237, 250, 251, 255, 256, 257, 258, 259, 262, 264, 267, 271, 272, 276, 277, 279, 280, 281, 282, 283],
  'mit': [2, 15, 16, 44, 21],
  'spider': [] + [4, 5, 49, 50, 175, 176, 187, 188, 191, 192, 211, 212, 213, 214, 215, 216, 217, 218, 223, 224, 225, 226, 227, 228, 229, 230, 237, 238, 239, 240, 241, 242, 243, 244, 283, 284, 387, 388, 445, 446, 544, 744, 776, 798, 799, 832, 833, 850, 851, 898, 899]
}

def group_metrics(dataset: str, metrics: np.ndarray, retrieval_score: bool=None):
  with open(f'{PREFIX}/data/{dataset}/queries.json') as f:
    sqls = [x['sql'] for x in json.load(f)]

  if retrieval_score:
    num_tables = [len(set(sql_to_tables(dataset, sql))) for i, sql in enumerate(sqls)]
  else:
    num_tables = [len(set(sql_to_tables(dataset, sql))) for i, sql in enumerate(sqls) if i not in SKIP_IDXS[dataset]]

  num_tables = np.array(num_tables)

  # print(f'#table: 2, metric: {metrics[num_tables == 2].mean(axis=0)}')

  metric_1 = 100*metrics[num_tables == 1].mean(axis=0)
  metric_2 = 100*metrics[num_tables == 2].mean(axis=0)
  metric_3 = 100*metrics[num_tables >= 3].mean(axis=0)

  print(f'#table: 1, metric: {sig_fig(metric_1)}')
  print(f'#table: 2, metric: {sig_fig(metric_2)}')
  print(f'#table: 3+, metric: {sig_fig(metric_3)}')

  # for num_table in range(min(num_tables), max(num_tables)+1):
  #   print(f'#table: {num_table}, metric: {metrics[num_tables == num_table].mean(axis=0)}')
