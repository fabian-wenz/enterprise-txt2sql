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
from scipy.optimize import linear_sum_assignment


class TimeoutError(Exception):
  pass

def best_matching(input_dict):
    # Extract row and column indices
    rows = sorted(input_dict.keys())  # Outer keys
    cols = sorted(next(iter(input_dict.values())).keys())  # Inner keys

    # Create a cost matrix (negative because we want to maximize scores)
    cost_matrix = np.zeros((len(rows), len(cols)))

    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            cost_matrix[i, j] = -input_dict[row][col]  # Negate for maximizing

    # Apply Hungarian algorithm to get the optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build the result: match the pairs
    matching = [(rows[i], cols[j]) for i, j in zip(row_ind, col_ind)]

    return matching

def handler(signum: int, frame: Any) -> None:
  raise TimeoutError("Calculation timed out")


class Metric:
  def __init__(self, gold_set: set, predicted_set: set):
    self.true_positives = len(gold_set.intersection(predicted_set))
    #print(len(gold_set))
    #print(len(predicted_set))

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
    self.db_config = db_config
    self.db_connection = None

  def initialize(self, eval_type: EvaluationType, gold_queries: List[str], gpt_queries: List[str]):
    if self.size != len(gold_queries) or self.size != len(gpt_queries):
      raise Exception("Sizes don't match!")
    if eval_type == EvaluationType.TABLE_NAME_EXTRACTION:
      self._calc_table_name_extraction(gold_queries, gpt_queries)
    elif eval_type == EvaluationType.VALUE_EXTRACTION:
      self._calc_table_name_extraction(gold_queries, gpt_queries)
      self._calc_value_extraction(gold_queries, gpt_queries)
      print('CA: ' + str(self.column_accuracy(gold_queries, gpt_queries)))
      print('EA: ' + str(self.entry_accuracy(gold_queries, gpt_queries)))
      print('OA: ' + str(self.output_accuracy(gold_queries, gpt_queries)))
      print('E%: ' + str(self.error_percentage(gpt_queries)))
    self._generate_metrics(eval_type)

  def get_metric_summary(self, eval_type: EvaluationType) -> (float, float, float, float):
    metrics = self.get_metrics(eval_type)
    #print([m.accuracy for m in metrics])
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

  def _generate_metrics(self, eval_type: EvaluationType):
    metrics = self.get_metrics(eval_type)
    if len(metrics) == 0:
      ground_truth, prediction = self.get_ground_truth_and_pred(eval_type)
      for i in range(self.size):
        metrics.append(Metric(ground_truth[i], prediction[i]))
    #print(metrics)

  def column_accuracy(self, gold_queries: List[str], gpt_queries: List[str]):
    CA_AVG = 0
    for i in range(self.size):
      try:
        predicted_columns_set = set([c.split('.')[1].upper() if '.' in c.upper() else c.upper() for c in Parser(gpt_queries[i]).columns])
        gold_columns_set = set([c.split('.')[1].upper() if '.' in c.upper() else c.upper() for c in Parser(gold_queries[i]).columns])

        # Intersection of columns
        correctly_identified_columns = gold_columns_set.intersection(predicted_columns_set)

        # Column Accuracy formula
        ca = len(correctly_identified_columns)*100 / len(gold_columns_set)
        CA_AVG += ca/self.size
      except Exception as e:

        print(e)
    return CA_AVG

  def error_percentage(self, queries: List[str]):
    errors = 0
    for i in range(len(queries)):
      hashed_gpt_sql = hashlib.sha256(queries[i].encode('utf-8')).hexdigest()
      file_name_gpt_sql = f'{PREFIX}/data/{db_id}/output/{hashed_gpt_sql}.pkl'  # .npy
      if self.db_connection is None:
          self.db_connection = DBConnection(self.db_config)
      if os.path.isfile(file_name_gpt_sql):
        with open(file_name_gpt_sql, "rb") as f:
          rows = pickle.load(f)
          if not rows:
            try:
              self.db_connection.execute_query(queries[i], self.tables_pred_sqls[i])
            except Exception as e:
              print(e)
              errors +=1
      else:
        try:
          self.db_connection.execute_query(queries[i], self.tables_pred_sqls[i])
        except Exception as e:
          print(e)
          errors +=1
    return errors/len(queries)*100
  def entry_accuracy(self, gold_queries: List[str], gpt_queries: List[str]):
    EA_AVG = 0
    for i in range(len(gold_queries)):
      hashed_gold_sql = hashlib.sha256(gold_queries[i].encode('utf-8')).hexdigest()
      file_name_gold_sql = f'{PREFIX}/data/{db_id}/output/{hashed_gold_sql}.pkl'  # .npy
      hashed_gpt_sql = hashlib.sha256(gpt_queries[i].encode('utf-8')).hexdigest()
      file_name_gpt_sql = f'{PREFIX}/data/{db_id}/output/{hashed_gpt_sql}.pkl'  # .npy
      if os.path.isfile(file_name_gold_sql):
        with open(file_name_gold_sql, "rb") as f:
          rows = pickle.load(f)
      else:
        if self.db_connection is None:
          self.db_connection = DBConnection(self.db_config)
        try:
          gold_output = self.db_connection.execute_query(gold_queries[i], self.tables_gold_sqls[i])
        except:
          print('gold')
          gold_output = []
        rows = set(gold_output)
        with open(file_name_gold_sql, "wb") as f:
          pickle.dump(rows, f)
      gold_tuple_of_lists = tuple(map(list, zip(*rows)))

      if os.path.isfile(file_name_gpt_sql):
        with open(file_name_gpt_sql, "rb") as f:
          rows = pickle.load(f)
      else:
        #print(gpt_queries[i])
        if self.db_connection is None:
          self.db_connection = DBConnection(self.db_config)
        try:
          gpt_output = self.db_connection.execute_query(gpt_queries[i], self.tables_pred_sqls[i])
        except:
          print('gpt')
          gpt_output = []
        rows = set(gpt_output)
        with open(file_name_gpt_sql, "wb") as f:
          pickle.dump(set(gpt_output), f)
      pred_tuple_of_lists = tuple(map(list, zip(*rows)))

      if len(gold_tuple_of_lists) == 0 or len(pred_tuple_of_lists) == 0:
        print('FAIL')
        continue
      min_list = gold_tuple_of_lists
      max_list = pred_tuple_of_lists
      gold_is_min = True
      if min(len(gold_tuple_of_lists), len(pred_tuple_of_lists)) == len(pred_tuple_of_lists):
        gold_is_min = False
        min_list = pred_tuple_of_lists
        max_list = gold_tuple_of_lists
      comp = {}
      print(gpt_queries[i])
      print(min_list)
      print(gold_queries[i])
      print(max_list)
      for i in range(len(min_list)):
        comp[i] = {}
        for j in range(len(max_list)):
          comp[i][j]=len(set(min_list[i]).intersection(set(max_list[j])))/(len(gold_tuple_of_lists[i] if gold_is_min else gold_tuple_of_lists[j]))
      matching = best_matching(comp)
      ea= 0
      for match in matching:
        ea += comp[match[0]][match[1]]*100/len(matching)
      EA_AVG += ea/self.size
    return EA_AVG

  def output_accuracy(self, gold_queries: List[str], gpt_queries: List[str]):
    OA_AVG = 0
    for i in range(len(gold_queries)):
      hashed_gold_sql = hashlib.sha256(gold_queries[i].encode('utf-8')).hexdigest()
      file_name_gold_sql = f'{PREFIX}/data/{db_id}/output/{hashed_gold_sql}.pkl'  # .npy
      hashed_gpt_sql = hashlib.sha256(gpt_queries[i].encode('utf-8')).hexdigest()
      file_name_gpt_sql = f'{PREFIX}/data/{db_id}/output/{hashed_gpt_sql}.pkl'  # .npy
      if os.path.isfile(file_name_gold_sql):
        with open(file_name_gold_sql, "rb") as f:
          gold_rows = pickle.load(f)
      else:
        if self.db_connection is None:
          self.db_connection = DBConnection(self.db_config)
        try:
          gold_output = self.db_connection.execute_query(gold_queries[i], self.tables_gold_sqls[i])
        except:
          print('gold')
          gold_output = []
        gold_rows = set(gold_output)
        with open(file_name_gold_sql, "wb") as f:
          pickle.dump(gold_rows, f)
      gold_tuple_of_lists = tuple(map(list, zip(*gold_rows)))

      if os.path.isfile(file_name_gpt_sql):
        with open(file_name_gpt_sql, "rb") as f:
          pred_rows = pickle.load(f)
      else:
        #print(gpt_queries[i])
        if self.db_connection is None:
          self.db_connection = DBConnection(self.db_config)
        try:
          gpt_output = self.db_connection.execute_query(gpt_queries[i], self.tables_pred_sqls[i])
        except:
          print('gpt')
          gpt_output = []
        pred_rows = set(gpt_output)
        with open(file_name_gpt_sql, "wb") as f:
          pickle.dump(set(gpt_output), f)
      pred_tuple_of_lists = tuple(map(list, zip(*pred_rows)))

      if len(gold_tuple_of_lists) == 0 or len(pred_tuple_of_lists) == 0:
        continue
      min_list = gold_tuple_of_lists
      max_list = pred_tuple_of_lists
      gold_is_min = True
      if min(len(gold_tuple_of_lists), len(pred_tuple_of_lists)) == len(pred_tuple_of_lists):
        gold_is_min = False
        min_list = pred_tuple_of_lists
        max_list = gold_tuple_of_lists
      comp = {}
      for i in range(len(min_list)):
        comp[i] = {}
        for j in range(len(max_list)):
          comp[i][j]=len(set(min_list[i]).intersection(set(max_list[j])))/(len(gold_tuple_of_lists[i] if gold_is_min else gold_tuple_of_lists[j]))
      matching = best_matching(comp)
      ml= []
      for match in matching:
        ml.append(max_list[match[1]])
      max_list = tuple(ml)
      OA_AVG += len(set(zip(*min_list)).intersection(set(zip(*max_list))))*100/(len(gold_rows)*self.size)
    return OA_AVG


  def _calc_value_extraction(self, gold_queries: List[str], gpt_queries: List[str]):
    if len(self.results_gold_sqls) == 0 and len(self.results_pred_sqls) == 0:
      for i in range(len(gold_queries)):
        #print(i)
        hashed_gold_sql = hashlib.sha256(gold_queries[i].encode('utf-8')).hexdigest()
        file_name_gold_sql = f'{PREFIX}/data/{db_id}/output/{hashed_gold_sql}.pkl'  # .npy
        hashed_gpt_sql = hashlib.sha256(gpt_queries[i].encode('utf-8')).hexdigest()
        file_name_gpt_sql = f'{PREFIX}/data/{db_id}/output/{hashed_gpt_sql}.pkl'  # .npy
        if os.path.isfile(file_name_gold_sql):
          # start_time = time.time()
          with open(file_name_gold_sql, "rb") as f:
            self.results_gold_sqls.append(pickle.load(f))
          # end_time = time.time()
          # elapsed_time = end_time - start_time
          # print(f'{i}: {elapsed_time}')
        else:
          if self.db_connection is None:
            self.db_connection = DBConnection(self.db_config)
          try:
            #  # start_time = time.time()
            gold_output = self.db_connection.execute_query(gold_queries[i], self.tables_gold_sqls[i])
          #  # end_time = time.time()
          #  # elapsed_time = end_time - start_time
          #  # print(f'{i}: {elapsed_time}')
          except Exception as e:
            print(e)
            print('gold')
            gold_output = []
          self.results_gold_sqls.append(set(gold_output))
          with open(file_name_gold_sql, "wb") as f:
            pickle.dump(set(gold_output), f)
        if os.path.isfile(file_name_gpt_sql):
          with open(file_name_gpt_sql, "rb") as f:
            self.results_pred_sqls.append(pickle.load(f))
        else:
          #print(gpt_queries[i])
          if self.db_connection is None:
            self.db_connection = DBConnection(self.db_config)
          try:
            if i not in []:
              gpt_output = self.db_connection.execute_query(gpt_queries[i], self.tables_pred_sqls[i])
            else:
              gpt_output = []
          except:
            print('gpt')
            gpt_output = []
          self.results_pred_sqls.append(set(gpt_output))
          with open(file_name_gpt_sql, "wb") as f:
            pickle.dump(set(gpt_output), f)

  def _calc_table_name_extraction(self, gold_queries: List[str], gpt_queries: List[str]):
    if len(self.tables_gold_sqls) == 0 and len(self.tables_pred_sqls) == 0:
      print(gold_queries)
      for i in range(len(gold_queries)):
        self.tables_gold_sqls.append(set([t.upper() for t in Parser(gold_queries[i]).tables]))
        try:
          self.tables_pred_sqls.append(set([t.upper() for t in Parser(gpt_queries[i]).tables]))
        except Exception as e:
          #print(str(e))
          self.tables_pred_sqls.append(set())


def _reformat_for_oracle(query: str, tables: set[str]):
  if tables is None:
    tables = set()
  # add wareuser for queries which were not being found
  for table in tables:
    table_mod = ' WAREUSER.' + table
    query = query.replace(' ' + table, table_mod)
  return query

def _reformat_for_tig(query: str, tables: set[str]):
  if tables is None:
    tables = set()
  # add wareuser for queries which were not being found
  for table in tables:
    table_mod = f" '{DB_DBID[1:]}-" + table + "'"
    #query = query.replace(' ' + table, table_mod)
    query = re.sub(r'(?<=[\s\.\(])' + table + r'(?=[\s\.\)])', table_mod, query)
  return query

class DBConnection:
  def __init__(self, db_config: Dict[str, str]):
    self.db_type: str = db_config['type']
    if self.db_type == 'oracle':
      self.connection = cx_Oracle.connect(user=CONFIG['username'], password=CONFIG['password'], dsn=CONFIG['dsn'])
      # Set input type handler to cx_Oracle.STRING
      self.connection.inputtypehandler = cx_Oracle.STRING
    elif self.db_type == 'sqlite3':
      self.connection = sqlite3.connect(f'{PREFIX}/data/{db_id}/database/{db_config["db_file"]}')
    else:
      raise ValueError("Unsupported database type")

  def __del__(self):
    self.connection.close()

  def _preprocess_query(self, query: str) -> str:
    if db_id == 'tig':
      query = _reformat_for_tig(query, set(Parser(query).tables))
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
