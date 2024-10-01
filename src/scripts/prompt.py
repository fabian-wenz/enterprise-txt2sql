from enum import Enum
import os
from pathlib import Path

import numpy as np
import pandas as pd
from collections import deque
from typing import *

import os
import sys

from sql_metadata import Parser

sys.path.append(os.path.abspath('./'))

from src.scripts.constants import *


# def detect_primary_keys():
#  for

def create_table_statement(table_name: str, path_to_schema: str = f'/data/{db_id}/schema{DB_DBID}/'):
  # TODO: Maybe add here desc, primary_keys
  if db_id == 'tig':
    table_name = table_name.lower()
  schema_file = PREFIX + path_to_schema + table_name + '.csv'
  if os.path.isfile(schema_file):
    df_t = pd.read_csv(schema_file)
    _schema_info, additional_info = '', ''
    schema_info = "CREATE TABLE " + table_name + "(\n"
    for i in range(len(df_t)):
      if PRIM_KEYS_ACTIVE and type(df_t['PKEY'][i]) == str and 'PRIMARY KEY' == df_t['PKEY'][i]:
          additional_info = f'  PRIMARY KEY ({df_t["COLUMN_NAME"][i]}),\n' + additional_info
      if PRIM_KEYS_ACTIVE and type(df_t['FKEY'][i]) == str and 'FOREIGN KEY' in df_t['FKEY'][i]:
          additional_info += f'  FOREIGN KEY({df_t["COLUMN_NAME"][i]}){df_t["FKEY"][i][12:]}\n'
      _schema_info += f'  {df_t["COLUMN_NAME"][i]} {df_t["DATA_TYPE"][i]},\n'
    if len(additional_info) > 0:
      schema_info += _schema_info + additional_info[:-1]
    else:
      schema_info += _schema_info[:-2]
    schema_info += '\n)\n\n'
    return schema_info
  else:
    raise Exception("File does not exist!")


'''
class TablePrompt:
  def __init__(self, table_name: str, logs: List[Tuple[str, str]]):
    self._table_name: str = table_name
    self._schema: str = create_table_statement(table_name)
    self._logs: List[(str, str)] = logs  # first: questions, second sqls
    # TODO: Maybe add here desc, primary_keys and adjust the tostring method

  def to_string(self, query: str, sql_logs: bool, question_logs: bool, num_of_examples: int,
                used_queries: List[str]) -> str:
    prompt = self._schema
    if sql_logs and num_of_examples > 0 and len(self._logs) > 0 and \
        (len(used_queries) == 0 or not all([l[1] in used_queries for l in self._logs])):
      if question_logs:
        prompt += "\n ** PREVIOUS QUESTIONS AND CORRESPONDING SQL QUERIES THAT REFERRED TO THE TABLE {0} **\n". \
          format(self._table_name)
      else:
        prompt += "\n ** PREVIOUS SQL QUERIES THAT REFERRED TO THE TABLE {0} **\n".format(self._table_name)
      for i in range(min(num_of_examples, len(self._logs))):
        if self._logs[i][1] != query and not self._logs[i][1] in used_queries:
          if question_logs:
            prompt += "Question: {0} \n".format(self._logs[i][0])
          prompt += "SQL: {0} \n".format(self._logs[i][1])
          used_queries.append(self._logs[i][1])
      num_of_examples -= len(self._logs)
    return prompt
'''


class ExamplePrompt:
  def __init__(self, question_: str, query_: str):
    self.question: str = question_
    self.query: str = query_
    self.involved_tables: List[str] = Parser(self.query.upper()).tables

  def to_string(self) -> str:
    # prompt = " ** PSEUDO EXAMPLE NR. {0} ** \n".format(example_num)
    prompt = ''
    for table in self.involved_tables:
      prompt += create_table_statement(table)
    prompt += "Question: {0} \n".format(self.question)
    prompt += "Answer: {0} \n".format(self.query)
    prompt += '\n\n'
    return prompt


class Prompt:
  def __init__(self, questions_: List[str], sqls_: List[str], examples_: List[Dict[str, str]]):
    if len(questions_) != len(sqls_):
      raise Exception("Number of questions and answers don't add up!")
    self.questions: List[str] = questions_
    self.sqls: List[str] = sqls_
    self.examples: List[ExamplePrompt] = [ExamplePrompt(examples_[i]['question'], examples_[i]['sql']) for i in range(len(examples_))]
    self.tables: List[str] = list(set([t for query in self.sqls for t in Parser(query.upper()).tables]))

  def get_prompt(self, query: str, top_k_tables: set[str], example_indices: List[int]) -> str:
    prompt = INITIAL_PROMPT
    if len(example_indices) > 0:
      #prompt += " ** {0}-SHOT PSEUDO_EXAMPLE ** \n".format(len(example_indices))
      for i in range(len(example_indices)):
        if self.examples[example_indices[i]].query != query:
          prompt += self.examples[example_indices[i]].to_string()
      prompt += "\n"
    if len(top_k_tables) > 0:
      #prompt += " ** TOP {0} MOST RELEVANT TABLES ** \n".format(len(top_k_tables))
      for table in top_k_tables:
        prompt += create_table_statement(table.upper())
      prompt += "\n"

    prompt += f"Question: {query}\n"
    prompt += f'Answer:'

    return prompt
