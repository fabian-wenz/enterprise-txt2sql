import hashlib
import signal
import time
from typing import *

import cx_Oracle
import re
from sql_metadata import Parser
import sqlite3
import numpy as np
import pickle
import threading

import os
import sys

sys.path.append(os.path.abspath('./'))

from src.scripts.constants import *


class TimeoutError(Exception):
  pass


def handler(signum: int, frame: Any) -> None:
  raise TimeoutError("Calculation timed out")


class Metric:
  def __init__(self, gold_set: set, predicted_set: set):
    self.true_positives = len(gold_set.intersection(predicted_set))
    print(len(gold_set))
    print(len(predicted_set))

    self.false_positives = len(predicted_set - gold_set)
    self.false_negatives = len(gold_set - predicted_set)
    self.precision = self.calc_precision()
    self.recall = self.calc_recall()
    self.accuracy = (gold_set == predicted_set)
    self.f1_score = self.calc_f1_score()

  def calc_precision(self) -> float:
    return self.true_positives / (self.true_positives + self.false_positives) \
      if (self.true_positives + self.false_positives) != 0 else 1

  def calc_recall(self) -> float:
    return self.true_positives / (self.true_positives + self.false_negatives) \
      if (self.true_positives + self.false_negatives) != 0 else 0

  def calc_f1_score(self) -> float:
    if self.recall is None or self.precision is None:
      raise Exception("Recall or Precision have not been defined yet.")

    return 2 * (self.precision * self.recall) / (self.precision + self.recall) \
      if (self.precision + self.recall) != 0 else 0

  def get_scores(self) -> List[float]:
    return [self.accuracy, self.precision, self.recall, self.f1_score]


class Evaluator:
  def __init__(self, size: int, db_config: Dict[str, str]):
    self.size = size
    if self.size == 0:
      raise Exception("Size cannot be 0!")
    self.tables_gold_sqls: List[set[str]] = []
    self.tables_pred_sqls: List[set[str]] = []
    self.results_gold_sqls: List[set] = []
    self.results_pred_sqls: List[set] = []
    self.table_metrics: List[Metric] = []
    self.precise_value_metric: List[Metric] = []
    self.db_connection = DBConnection(db_config)

  def initialize(self, eval_type: EvaluationType, gold_queries: List[str], gpt_queries: List[str]):
    if self.size != len(gold_queries) or self.size != len(gpt_queries):
      raise Exception("Sizes don't match!")
    if eval_type == EvaluationType.TABLE_NAME_EXTRACTION:
      self._calc_table_name_extraction(gold_queries, gpt_queries)
    elif eval_type == EvaluationType.VALUE_EXTRACTION:
      self._calc_table_name_extraction(gold_queries, gpt_queries)
      self._calc_value_extraction(gold_queries, gpt_queries)
    self.generate_metrics(eval_type)

  def get_metric_summary(self, eval_type: EvaluationType) -> (float, float, float, float):
    metrics = self.get_metrics(eval_type)
    print([m.accuracy for m in metrics])
    accuracy_avg = sum([m.accuracy for m in metrics]) / self.size
    recall_avg = sum([m.recall for m in metrics]) / self.size
    precision_avg = sum([m.precision for m in metrics]) / self.size
    f1_score_avg = sum([m.f1_score for m in metrics]) / self.size
    return accuracy_avg, recall_avg, precision_avg, f1_score_avg

  def get_metrics(self, eval_type: EvaluationType) -> List[Metric]:
    if eval_type == EvaluationType.TABLE_NAME_EXTRACTION:
      return self.table_metrics
    elif eval_type == EvaluationType.VALUE_EXTRACTION:
      return self.precise_value_metric

  def get_ground_truth_and_pred(self, eval_type: EvaluationType) -> (List[set], List[set]):
    if eval_type == EvaluationType.TABLE_NAME_EXTRACTION:
      return self.tables_gold_sqls, self.tables_pred_sqls
    elif eval_type == EvaluationType.VALUE_EXTRACTION:
      return self.results_gold_sqls, self.results_pred_sqls

  def generate_metrics(self, eval_type: EvaluationType):
    metrics = self.get_metrics(eval_type)
    if len(metrics) == 0:
      ground_truth, prediction = self.get_ground_truth_and_pred(eval_type)
      for i in range(self.size):
        metrics.append(Metric(ground_truth[i], prediction[i]))
    print(metrics)

  def _calc_value_extraction(self, gold_queries: List[str], gpt_queries: List[str]):
    if len(self.results_gold_sqls) == 0 and len(self.results_pred_sqls) == 0:
      for i in range(len(gold_queries)):
        print(i)
        hashed_gold_sql = hashlib.sha256(gold_queries[i].encode('utf-8')).hexdigest()
        file_name_gold_sql = f'{prefix}/data/{db_id}/output/{hashed_gold_sql}.pkl'  # .npy
        hashed_gpt_sql = hashlib.sha256(gpt_queries[i].encode('utf-8')).hexdigest()
        file_name_gpt_sql = f'{prefix}/data/{db_id}/output/{hashed_gpt_sql}.pkl'  # .npy
        if os.path.isfile(file_name_gold_sql):
          # start_time = time.time()
          with open(file_name_gold_sql, "rb") as f:
            self.results_gold_sqls.append(pickle.load(f))
          # end_time = time.time()
          # elapsed_time = end_time - start_time
          # print(f'{i}: {elapsed_time}')
        else:
          try:
          #  # start_time = time.time()
            gold_output = self.db_connection.execute_query(gold_queries[i], self.tables_gold_sqls[i])
          #  # end_time = time.time()
          #  # elapsed_time = end_time - start_time
          #  # print(f'{i}: {elapsed_time}')
          except:
            print('gold')
            gold_output = []
          self.results_gold_sqls.append(set(gold_output))
          with open(file_name_gold_sql, "wb") as f:
            pickle.dump(set(gold_output), f)
        if os.path.isfile(file_name_gpt_sql):
          with open(file_name_gpt_sql, "rb") as f:
            self.results_pred_sqls.append(pickle.load(f))
        else:
          print(gpt_queries[i])
          try:
            gpt_output = self.db_connection.execute_query(gpt_queries[i], self.tables_pred_sqls[i])
          except:
            print('gpt')
            gpt_output = []
          self.results_pred_sqls.append(set(gpt_output))
          with open(file_name_gpt_sql, "wb") as f:
            pickle.dump(set(gpt_output), f)

  def _calc_table_name_extraction(self, gold_queries: List[str], gpt_queries: List[str]):
    if len(self.tables_gold_sqls) == 0 and len(self.tables_pred_sqls) == 0:
      for i in range(len(gold_queries)):
        self.tables_gold_sqls.append(set([t.upper() for t in Parser(gold_queries[i]).tables]))
        try:
          self.tables_pred_sqls.append(set([t.upper() for t in Parser(gpt_queries[i]).tables]))
        except Exception as e:
          print(str(e))
          self.tables_pred_sqls.append(set())


def _reformat_for_oracle(query: str, tables: set[str]):
  if tables is None:
    tables = set()
  # add wareuser for queries which were not being found
  for table in tables:
    table_mod = ' WAREUSER.' + table
    query = query.replace(' ' + table, table_mod)
  return query


class DBConnection:
  def __init__(self, db_config: Dict[str, str]):
    self.db_type: str = db_config['type']
    if self.db_type == 'oracle':
      self.connection = cx_Oracle.connect(user=CONFIG['username'], password=CONFIG['password'], dsn=CONFIG['dsn'])
      # Set input type handler to cx_Oracle.STRING
      self.connection.inputtypehandler = cx_Oracle.STRING
    elif self.db_type == 'sqlite3':
      self.connection = sqlite3.connect(f'{prefix}/data/{db_id}/database/{db_config["db_file"]}')
    else:
      raise ValueError("Unsupported database type")

  def __del__(self):
    self.connection.close()

  def _preprocess_query(self, query: str) -> str:
    if query[-1] == ';':
      query = query[:-1]
    if 'LIMIT' in query and self.db_type == 'oracle':
      contains_limit = True
      limit_number = re.search(r'\bLIMIT\s+(\d+)', query)
      limit_value = int(limit_number.group(1))
      query = query[:query.index('LIMIT ')]
      query = 'SELECT * FROM (' + query + ') WHERE ROWNUM <=' + str(limit_value)

    # if query exceed line limit
    if len(query) > 3000:
      query_temp = ''
      last_space_index = query[:3000].rfind(' ')
      i = 0
      while i < len(query):
        query_temp += query[i:i + last_space_index] + '\n'
        i += last_space_index
        if last_space_index + 3000 > len(query):
          last_space_index = len(query) - i
        else:
          last_space_index = query[i:i + 3000].rfind(' ')
      query = query_temp[:-1]
    return query

  def execute_query(self, query: str, tables: Set[str] = None) -> List:
    query = self._preprocess_query(query)

    # Create a cursor object
    cursor = self.connection.cursor()

    # Set DEFINE OFF
    cursor.setoutputsize(0)
    # Execute SQL queries
    try:
      cursor.execute(query)
      # Fetch the results
      rows = cursor.fetchall()
    except Exception as e1:
      error_message = str(e1)
      if self.db_type == "oracle":
        query = _reformat_for_oracle(query, tables)
        try:
          cursor.execute(query)
          # Fetch the results
          rows = cursor.fetchall()
          cursor.close()
          return rows
        except Exception as e2:
          # Print the error message to stderr
          print("An error occurred:", str(e1))
          print("An error occurred:", str(e2))
          error_message = "First error without replacing tables: {} \n Second Error when replacing tables: {}.".format(
            error_message,
            str(e2))
      cursor.close()
      print('___________')
      print(query)
      print('___________')
      raise Exception(error_message)
    cursor.close()
    return rows
