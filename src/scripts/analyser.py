import re
from statistics import mean
from typing import Tuple
import os
import sys

import sqlparse

sys.path.append(os.path.abspath('./'))
import json

from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sql_metadata import Parser
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from sqlglot import parse_one, exp, transpile
from tqdm import tqdm
from typing import List

from src.scripts.constants import PREFIX
from src.scripts.utils import sql_to_tables, sig_fig

sql_keywords = {'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'WHERE', 'JOIN', 'LEFT', 'RIGHT', 'FULL', 'INNER',
                'OUTER', 'GROUP', 'ORDER', 'BY', 'HAVING', 'DISTINCT', 'CREATE', 'ALTER', 'DROP', 'INDEX', 'UNION',
                'ALL', 'VALUES', 'INTO', 'AS', 'SET', 'AND', 'OR', 'NOT', 'ON', 'LIMIT', 'OFFSET', 'LIKE', 'IN',
                'BETWEEN', 'IS', 'NULL', 'PRIMARY', 'FOREIGN', 'KEY', 'REFERENCES', 'EXISTS', 'CAST', 'CASE', 'WHEN',
                'THEN', 'ELSE', 'END', 'EXCEPT', 'INTERSECT', 'TABLE', 'VIEW', 'TRIGGER', 'PROCEDURE', 'FUNCTION',
                'DATABASE', 'USE', 'GRANT', 'REVOKE', 'WITH', 'REPLACE', 'CHECK', 'DEFAULT', 'CONSTRAINT', 'TRUNCATE',
                'COMMIT', 'ROLLBACK', 'SAVEPOINT', 'TRANSACTION', 'SEQUENCE', 'CURSOR', 'FETCH', 'DECLARE', 'OPEN',
                'CLOSE', 'LOOP', 'WHILE', 'FETCH', 'MERGE', 'MINUS', 'RETURNING', 'SYNONYM', 'START', 'BEGIN', 'END',
                'ELSEIF', 'PACKAGE', 'PARTITION', 'RENAME', 'EXPLAIN', 'ANALYZE', 'LOG', 'READ', 'ONLY', 'LOCK',
                'SHARE', 'GRANT', 'USER', 'PASSWORD', 'EXCLUSIVE', 'UNIQUE', 'ROWNUM', 'INTERVAL', 'CURRENT_DATE',
                'CURRENT_TIMESTAMP', 'SYSDATE', 'SYSTIMESTAMP', 'SESSION', 'SYSTEM', 'LEVEL', 'RANK', 'DENSE_RANK',
                'ROW_NUMBER', 'PERCENT_RANK', 'CUME_DIST', 'NTILE', 'LAG', 'LEAD', 'FIRST_VALUE', 'LAST_VALUE',
                'LISTAGG', 'GROUPING', 'CUBE', 'ROLLUP', 'ORDER BY', 'PARTITION BY', 'OVER', 'WINDOW', 'RANGE', 'ROWS',
                'UNBOUNDED PRECEDING', 'CURRENT ROW', 'NULLS FIRST', 'NULLS LAST'}
def find_sum_and_max(arr: [int]) -> Tuple[int, int]:
  return (sum(arr), max(arr)) if arr else (0, 0)


def normalizer(node):
    # if isinstance(node, exp.Column):
    #     return parse_one("PLACEHOLDER_COLUMN")
    # elif isinstance(node, exp.Table):
    #     return parse_one("PLACEHOLDER_TABLE")
    if isinstance(node, exp.Literal):
        return parse_one("PLACEHOLDER_LITERAL")
    # elif type(node) in [
    #     exp.Count,
    #     exp.Sum,
    #     exp.Min,
    #     exp.Max,
    #     exp.Avg,
    #     exp.Quantile,
    #     exp.Stddev,
    #     exp.StddevPop,
    #     exp.StddevSamp,
    # ]:
    #     return parse_one("PLACEHOLDER_AGG")
    # elif type(node) in [exp.LT, exp.LTE, exp.GT, exp.GTE]:
    #     return parse_one("PLACEHOLDER_COMPARISON")
    return node


def noramlize_sql(parsed_sql):
  return parsed_sql.transform(normalizer).sql()

class Analyser:
  def __init__(self, dataset:str, analyzing_file: str):
    # Initialize keywords and regex patterns
    self.keywords = {
        'selection': 'SELECT',
        'filtering': r'^(?=.*(?:WHERE|HAVING)(?:.*(?:=|>|<|IN|BETWEEN|AND|OR|NOT|LIKE)))',
        'aggregation': r'\b(?:SUM|AVG|MAX|MIN)\b',
        'counting': 'COUNT',
        'summing': 'SUM',
        'min': 'MIN',
        'max': 'MAX',
        'averaging':'AVG',
        'group_concat':'GROUP_CONCAT|STRING_AGG',
        'stats':'VARIANCE|STDEV|MEDIAN|PERCENTILE_CONT|PERCENTILE_DISC|COVAR_POP|COVAR_SMAP|CORR',
        'grouping': 'GROUP BY',
        'ordering': 'ORDER BY',
        'window': 'OVER',
        'top-k': r'^(?=.*ORDER BY )(?=.*LIMIT )',
        'join': 'JOIN',
        'intersect': 'INTERSECT',
        'union': 'UNION',
        'distinct': 'DISTINCT',
        'pattern_matching': 'LIKE',
        'nested': r'^(?=.*SELECT.*\(SELECT)'
    }
    self.regex_patterns = {key: re.compile(pattern) for key, pattern in self.keywords.items()}
    self.df_q = pd.read_json(analyzing_file)
    #self.df_q = pd.read_csv(analyzing_file)
    #self.df_q['sql'] = self.df_q['prediction']
    self.dataset = dataset

  def reload(self, analyzing_file: str):
    self.df_q = pd.read_json(analyzing_file)

  def analyze(self):
    print(f'{self.dataset}:')
    keys = self.count_keywords()
    toks = self.count_toks()
    n_grams = self.count_total_n_grams()
    tables = self.count_tables()
    columns = self.count_query_columns()
    print(f'AVG keywords: {sum([value[0] for value in keys.values()])/len(self.df_q)}')
    print(f'AVG tokens: {toks[0]}')
    print(f'AVG 3-grams: {n_grams[0]}')
    print(f'AVG used tables: {tables[0]}')
    #print(f'MAX used tables: {tables[1]}')
    print(f'AVG used columns: {columns[0]}')
    print(f"AVG used nesting: {self.count_given(['selection'])}")
    print(f"AVG used agg: {self.count_given(['min', 'max', 'counting', 'summing', 'averaging', 'window', 'stats', 'grouping', 'ordering'])}")
    self.keywords = {'window': ' OVER '}
    self.regex_patterns = {key: re.compile(pattern) for key, pattern in self.keywords.items()}
    keys = self.count_keywords()
    print(f'AVG keywords: {sum([value[0] for value in keys.values()])/len(self.df_q)}')
    print('___________________________________________________________________________________')
    print(self.count_sql_keywords())
    print(len(self.df_q))
  def count_sql_keywords(self):
    print(len(sql_keywords))
    keyword_counts = Counter()
    for query in self.df_q['sql']:
      # Parse the query
      parsed = sqlparse.parse(query)[0]

      # Initialize a counter to store keyword occurrences

      # Iterate over tokens to check for keywords
      for sql_keyword in sql_keywords:
          if sql_keyword in query:
              keyword_counts[sql_keyword] = 1

    return sum(keyword_counts.values())
  def count_given(self, keys:list[str]):
    aggregations = {category: [] for category in keys}#, 'ordering', 'window']}
    for category in aggregations:
      for sql in self.df_q['sql']:
        count = len(re.findall(self.regex_patterns[category], sql))
        aggregations[category].append(count)

    return sum([sum(counts) for category, counts in aggregations.items()])

  def count_keywords(self):
    keyword_counts = {category: [] for category in self.keywords}
    for category, pattern in self.regex_patterns.items():
      for sql in self.df_q['sql']:
        count = len(re.findall(pattern, sql))
        keyword_counts[category].append(count)

    return {category: find_sum_and_max(counts) for category, counts in keyword_counts.items()}

  def count_toks(self):
    toks = []
    for sql in self.df_q['sql']:
      toks.append(len(word_tokenize(sql)))
    return sum(toks) / len(self.df_q), max(toks)

  def count_n_grams_per_query(self, n: int = 3):
    n_grams = []
    for sql in self.df_q['sql']:
      toks = word_tokenize(sql)
      print(toks)
      [n_grams.append(' '.join(toks[i:(i+n)])) for i in range(len(toks)-(n-1))]
    return len(set(n_grams)) / len(self.df_q)
  def count_total_n_grams(self, n: int = 3):
    n_gram_counts = set()
    for sql in self.df_q['sql']:
      toks = word_tokenize(sql)
      # Generate all 3-grams
      n_grams = set(ngrams(toks, n))
      # Count the number of 3-grams
      n_gram_counts = n_gram_counts | n_grams
    return len(n_gram_counts) / len(self.df_q), max(n_gram_counts)

  #def count_tables(self):
  #  tables_nums = []
  #  for sql in self.df_q['sql']:
  #    tables_nums.append(len(sql_to_tables(sql)))
  #  print(dict(sorted(Counter(tables_nums).items())))
  #  return sig_fig(sum(tables_nums) / len(self.df_q)), sig_fig(max(tables_nums))
  def count_tables(self):
    tables_nums = []
    for sql in self.df_q['sql']:
      tables_nums.append(len(set(Parser(sql.upper()).tables)))
    return sum(tables_nums) / len(self.df_q), max(tables_nums)

  def count_query_columns(self):
    column_nums = []
    num_queries = len(self.df_q)
    for sql in self.df_q['sql']:
      try:
        column_nums.append(len(set(Parser(sql.upper()).columns)))
      except Exception as e:
        num_queries -= 1
    return sum(column_nums) / num_queries, max(column_nums)

  def count_db_stats(self):
    df = pd.read_csv(f'{PREFIX}/data/{self.dataset}/tables.csv')
    num_cols = [len(x.split(',')[1:]) for x in df['schema'].tolist()]
    num_cols_per_table = sig_fig(np.average(np.array(num_cols)))
    num_tables = df.shape[0]
    num_dbs = len(set(df['db_ids'].tolist()))
    num_tables_per_db = sig_fig(num_tables/num_dbs)
    print(f'#DBs: {num_dbs}, #Tables/DB: {num_tables_per_db}, #Cols/DB: {num_cols_per_table}')

def count_query(dataset: str, sqls: List[str]):
  error_cnt = 0
  cnts = {'join': [], 'aggregation': [], 'nesting': []}
  agg = []
  idxs = []
  for i, sql in enumerate(tqdm(sqls)):
    sql = sql.replace('`', '"')
    # print(sql)
    try:
      transpile(sql)
    except Exception as e:
      print(e)
      error_cnt += 1
      continue
    parsed_query = parse_one(sql)

    # count number of joins
    join_cnt = 0
    join_cnt += sum(1 for _ in parsed_query.find_all(exp.Join))
    if dataset == 'mit':
      for x in parsed_query.find_all(exp.Where):
        x = str(x).lower().replace('where', '')
        if 'and' in x:
          x = x.split('and')
        else:
          x = [x]
        print(x)
        join_cnt += len([_x for _x in x if '\'' not in _x and ' = ' in _x])
    # cnts['join'].append(join_cnt)

    # count number of nesting
    select_count = sum(1 for _ in parsed_query.find_all(exp.Select))
    intersect_count = sum(1 for _ in parsed_query.find_all(exp.Intersect))
    except_count = sum(1 for _ in parsed_query.find_all(exp.Except))
    nest_level = select_count - (intersect_count + except_count)
    # cnts['nesting'].append(nest_level)

    # count number of aggregations (SUM, AVG, MAX, MIN, COUNT, GROUP BY, PARTITION BY)
    agg_cnt = sum(sum(1 for _ in parsed_query.find_all(x)) for x in [exp.Sum, exp.Avg, exp.Max, exp.Min, exp.Count, exp.Group, exp.Partition])
    agg_cnt2 = sum(sql.lower().count(x) for x in ['sum(', 'avg(', 'min(', 'max(', 'count(', 'sum (', 'avg (', 'min (', 'max (', 'count (', 'group by', 'partition by'])
    # agg.append(agg_cnt2)
    # cnts['aggregation'].append(agg_cnt)

    if agg_cnt >= 1 or nest_level >= 2:
      cnts['join'].append(join_cnt)
      cnts['nesting'].append(nest_level)
      cnts['aggregation'].append(agg_cnt)
      agg.append(agg_cnt2)
      idxs.append(i)

  for k in cnts:
    cnts[k] = sig_fig(np.average(np.array(cnts[k])))
    print(k, cnts[k])
  print(f'agg2: {np.average(np.array(agg))}')
  print(f'error: {error_cnt}')

  return idxs

def count_redundancy(sqls: List[str]):
  error_cnt = 0
  distinct_sql_patterns = defaultdict(list)

  for idx, sql in enumerate(tqdm(sqls)):
    sql = sql.replace('`', '"')

    try:
      transpile(sql)
    except Exception as e:
      print(f"Could not transpile: {sql}\n{e}")
      error_cnt += 1
      continue

    parsed_query = parse_one(sql)
    # distinct_sql_patterns[noramlize_sql(parsed_query)].append(sql)
    distinct_sql_patterns[noramlize_sql(parsed_query)].append(idx)
    # max_queries_per_pattern = max(len(queries) for queries in distinct_sql_patterns.values())
    # standard_deviation_of_queries_per_pattern = numpy.std(
    #     [len(queries) for queries in distinct_sql_patterns.values()]
    # )

  # for k in distinct_sql_patterns:
  #   print(k)
  #   print('='*30)
  #   print(distinct_sql_patterns[k][0], end='\n\n')

  print(f'error: {error_cnt}')
  print(f'unique patterns: {len(distinct_sql_patterns)}')

  return distinct_sql_patterns

if __name__ == '__main__':
  dataset = 'fiben'
  a = Analyser(dataset, f'{PREFIX}/data/{dataset}/all_queries.json')
  #a = Analyser(dataset, f'{PREFIX}/data/results/_results_mit_top10_tables_samples_1_run_0_gpt-3.5-turbo-0125_(PRIM_KEYS_ACTIVE)_ERROR.csv')
  df = pd.read_csv(f'{PREFIX}/data/results/_results_mit_top10_tables_samples_1_run_0_gpt-3.5-turbo-0125_(PRIM_KEYS_ACTIVE)_ERROR.csv')
  print(df.keys())

  #with open(f'{PREFIX}/data/{dataset}/newest_queries.json') as f:
  #  sqls = [q['sql'] for q in json.load(f)]
  a.analyze()
  #print(a.count_db_stats())
  # a.count_query()
  #count_redundancy(sqls)
  #count_query('xxx', sqls)
