import argparse
from datetime import datetime

import pandas as pd
from sql_metadata import Parser
import json
from openai import OpenAI
from typing import List
import os
import pyperclip
# import tiktoken
import numpy as np

import os
import sys

sys.path.append(os.path.abspath('./'))

from src.scripts.prompt import Prompt
from src.scripts.retrieve import top_k
from src.scripts.constants import *
from src.scripts.eval import *


def get_gpt_preds(dataset: str, debug: bool, focus_q_idx: int | None, example_num: int):
  client = OpenAI(
    api_key=CONFIG['api_key']
  )

  resp_dict = []
  '''gpt_pred_fn = f'./model_preds/mit/{model}.json'

  if not debug:
    if os.path.isfile(gpt_pred_fn):
      with open(gpt_pred_fn, 'r') as f:
        resp_dict = json.load(f)'''

  with open(f'{PREFIX}/data/{dataset}/queries.json') as f:
    qs = json.load(f)

  with open(f'{PREFIX}/data/{dataset}/examples.json') as f:
    examples = json.load(f)

  prompt = Prompt([x['question'] for x in qs], [x['sql'] for x in qs], examples)

  for q_idx, q in enumerate(qs):
    if focus_q_idx is not None and q_idx != focus_q_idx:
      continue

    if f'{q_idx}' in resp_dict:
      continue

    if DB_DBID == 'csail_stata_nova':
      top_k_ = []
      for tk in top_k(dataset, q_idx, 20):
        if tk[:4] != 'api-':
          top_k_.append(tk)
      top_k_ = top_k_[:10]

      p = prompt.get_prompt(q['question'], set(top_k_), list(range(example_num)))
    else:
      p = prompt.get_prompt(q['question'], set(top_k(dataset, q_idx, K)), list(range(example_num)))
    #print(p)
    # len(p)
    #print(q_idx)
    #tokenizer = tiktoken.get_encoding("cl100k_base")
    #print(len(tokenizer.encode(p)))
    #print('__________')
    #print(p)
    #print('__________')

    if debug:
      print(p)
      # pyperclip.copy(p)
    else:
      resp = client.chat.completions.create(
        model=model,
        temperature=0,
        top_p=1.0,
        seed=42,
        messages=[
          {'role': 'user', 'content': p}
        ]
      )

      resp_content = resp.choices[0].message.content
      print(resp_content)
      if ERROR and q_idx != 31:
        print(q_idx)
        db = DBConnection({'type': db_type, 'db_file': DB_FILE})
        try:
          db.execute_query(resp_content, set(Parser(resp_content).tables))
        except Exception as e:
          print(e)
          p = prompt.get_prompt(q['question'], set(top_k(dataset, q_idx, K)), [])
          resp = client.chat.completions.create(
            model=model,
            temperature=0,
            top_p=1.0,
            seed=42,
            messages=[
              {'role': 'user', 'content': p + f"\n\nIn the first attempt it got for the following SQL:\n{resp_content}\n\nThe following error:{str(e)}\n Please correct the SQL and state only the new correct SQL-statement."}
            ]
          )
          resp_content = resp.choices[0].message.content

      # resp_dict[q_idx] = resp_content
      resp_dict.append(resp_content)

      # with open(gpt_pred_fn, 'w') as f:
      #  json.dump(resp_dict, f, indent=2)
  if not debug:
    key_active_string = ''
    if PRIM_KEYS_ACTIVE:
      key_active_string = '_(PRIM_KEYS_ACTIVE)'
    df = pd.DataFrame(
      data={'question': [x['question'] for x in qs], 'prediction': resp_dict, 'answer': [x['sql'] for x in qs]})
    run_index = 0
    resulting_file_name = PREFIX + f"/data/results/{DB_DBID[:]}{DATE_STRING}_results_{dataset}_top{K}_tables_samples_{example_num}_run_{run_index}_{model}{key_active_string}{'_ERROR' if ERROR else ''}.csv"
    while os.path.isfile(resulting_file_name):
      run_index += 1
      resulting_file_name = PREFIX +  f"/data/results/{DATE_STRING}_results_{dataset}_top{K}_tables_samples_{example_num}_run_{run_index}_{model}{key_active_string}{'_ERROR' if ERROR else ''}.csv"
    df.to_csv(resulting_file_name, index=False)
    print(df['prediction'])


def get_evaluation(dataset: str, example_num: int, num_of_avg_runs: int):
  df_s = []
  key_active_string = ''
  if PRIM_KEYS_ACTIVE:
    key_active_string = '_(PRIM_KEYS_ACTIVE)'
  for run_index in range(num_of_avg_runs):
    df_s.append(pd.read_csv(
      PREFIX +  f"/data/results/{DB_DBID[:]}{DATE_STRING}_results_{dataset}_top{K}_tables_samples_{example_num}_run_{run_index}_{model}{key_active_string}{'_ERROR' if ERROR else ''}.csv"))
    #df_s.append(pd.read_csv(
    #  PREFIX + '/data/results/{5}_results_{0}_top{1}_tables_samples_{2}_run_{3}_{4}{6}.csv'. \
    #  format(dataset, K, example_num, run_index, model, DATE_STRING, key_active_string)))

  majority_predicted = []
  for i in range(len(df_s[0])):
    occ = {}
    for j in range(len(df_s)):
      if df_s[j]['prediction'][i] in occ.keys():
        occ[df_s[j]['prediction'][i]] += 1
      else:
        occ[df_s[j]['prediction'][i]] = 1
    sorted_dict_desc = dict(sorted(occ.items(), key=lambda item: item[1], reverse=True))
    first_key = next(iter(sorted_dict_desc))
    majority_predicted.append(first_key)

  df_majority = pd.DataFrame(
    data={'question': df_s[0]['question'][:], 'prediction': majority_predicted[:], 'answer': df_s[0]['answer'][:]}).reset_index(drop=True)

  eval = Evaluator(len(df_majority), {'type': db_type, 'db_file': DB_FILE})

  eval.initialize(EVALUATION_TYPE, gold_queries=df_majority['answer'],
                  gpt_queries=df_majority['prediction'])
  metric_res = eval.get_metric_summary(EVALUATION_TYPE)
  print(eval.get_metric_summary(EVALUATION_TYPE))

  if os.path.isfile(PREFIX + '/data/metric-results.json'):
    with open(PREFIX + '/data/metric-results.json') as f:
      df = json.load(f)
  else:
    df = []

  pk = 'Y' if PRIM_KEYS_ACTIVE else 'N'
  pk += 'Y' if ERROR else 'N'

  new_row = {
    'date': DATE_STRING,
    'db_id': db_id,
    'model': model,
    'eval-type':'TABLE_NAME_EXTRACTION' if EVALUATION_TYPE is EvaluationType.TABLE_NAME_EXTRACTION else 'VALUE_EXTRACTION',
    'num-of-table-retrieval': K,
    'num-of-examples': example_num,
    'accuracy': metric_res[0],
    'recall': metric_res[1],
    'precision': metric_res[2],
    'f1-score': metric_res[3],
    'num-of-runs':num_of_avg_runs,
    'prim-and-freign-keys': pk
  }
  print(new_row)
  df.append(new_row)

  with open(PREFIX + '/data/metric-results.json', 'w') as f:
    json.dump(df, f, indent=2)


if __name__ == '__main__':
  #parser = argparse.ArgumentParser()
  #parser.add_argument("--database", type=str, choices=["spider", "mit", "bird", "fiben", ], default="spider")
  #parser.add_argument("--split", type=str, choices=["train", "test"], default="test",  required=True)
  #parser.add_argument("--k_shot", type=int, default=0, help="Number of examples")
  #args = parser.parse_args()
  #for i_ in [0, 1, 2, 4, 5]:
  # get_gpt_preds(db_id, debug=False, focus_q_idx=None, example_num=1)
  #K = 5
  #for i_ in [0, 1, 2, 4, 5]:
  #  get_gpt_preds(db_id, debug=False, focus_q_idx=None, example_num=1)
  #K = 0
  #for i_ in [0, 1, 2, 4, 5]:
  #  get_gpt_preds(db_id, debug=False, focus_q_idx=None, example_num=1)
  #K=10
  #for i_ in [0, 1, 2, 4, 5]:
  #  get_gpt_preds(db_id, debug=False, focus_q_idx=None, example_num=1)
  #K = 5
  #for i_ in [0, 1, 2, 4, 5]:
  #  get_gpt_preds(db_id, debug=False, focus_q_idx=None, example_num=1)
  #K = 0
  #for i_ in [0, 1, 2, 4, 5]:
  #  get_gpt_preds(db_id, debug=False, focus_q_idx=None, example_num=1)
  #  get_gpt_preds(db_id, debug=True, focus_q_idx=0, example_num=1)
  # for i_ in [0, 1, 2, 4, 5]:
  #  get_gpt_preds(db_id, debug=False, focus_q_idx=None, example_num=1)
  # for i_ in [0, 1, 2, 4, 5]:
  #  get_gpt_preds(db_id, debug=False, focus_q_idx=None, example_num=5)
  #K=10
  #get_gpt_preds(db_id, debug=False, focus_q_idx=None, example_num=1)
  get_evaluation(db_id, 1, 1)
  # K=5
  # get_evaluation(db_id, 1, 5)
  # K=0
  # get_evaluation(db_id, 1, 5)
  # K += 5
  # get_evaluation(db_id)
  # K += 5
  # get_evaluation(db_id)
