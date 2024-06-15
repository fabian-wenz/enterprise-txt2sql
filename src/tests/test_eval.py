import unittest

import os
import sys

sys.path.append(os.path.abspath('./'))
from src.scripts.eval import *


class TestEval(unittest.TestCase):
  def test_empty_Evaluator(self):
    try:
      Evaluator(0, {'type': 'oracle'})
    except Exception as error:
      self.assertEqual(str(error), "Size cannot be 0!")

  def test_mismatching_sizes(self):
    try:
      e = Evaluator(1, {'type': 'oracle'})
      e.initialize(EvaluationType.TABLE_NAME_EXTRACTION,
                   ["SELECT * FROM ACADEMIC_TERMS;", "SELECT * FROM ACADEMIC_TERMS;"],
                   ["SELECT * FROM ACADEMIC_TERMS_ALL;", "SELECT * FROM ACADEMIC_TERMS;"])
    except Exception as error:
      self.assertEqual(str(error), "Sizes don't match!")

  def test_metric_calculation(self):
    e = Evaluator(2, {'type': 'oracle'})
    self.assertEqual(len(e.tables_pred_sqls), 0)
    self.assertEqual(len(e.tables_gold_sqls), 0)
    self.assertEqual(len(e.table_metrics), 0)
    e.initialize(EvaluationType.TABLE_NAME_EXTRACTION,
                 ["SELECT * FROM ACADEMIC_TERMS;", "SELECT * FROM ACADEMIC_TERMS;"],
                 ["SELECT * FROM ACADEMIC_TERMS_ALL;", "SELECT * FROM ACADEMIC_TERMS;"])
    self.assertEqual(len(e.tables_pred_sqls), 2)
    self.assertEqual(len(e.tables_gold_sqls), 2)
    self.assertEqual(len(e.table_metrics), 2)
    results = e.get_metric_summary(EvaluationType.TABLE_NAME_EXTRACTION)
    self.assertEqual(results, (0.5, 0.5, 0.5, 0.5))


class TestMetric(unittest.TestCase):
  def test_metric(self):
    m = Metric({"a", "b"}, {"c", "b"})
    self.assertEqual(m.accuracy, 0.5)
    self.assertEqual(m.recall, 0.5)
    self.assertEqual(m.precision, 0.5)
    self.assertEqual(m.f1_score, 0.5)


class TestDBConnection(unittest.TestCase):
  def test_connection(self):
    db = DBConnection({'type': 'oracle'})
    self.assertEqual(len(db.execute_query("SELECT *  FROM ALL_TAB_COLUMNS WHERE ROWNUM <= 1;")), 1)

  def test_oracle(self):
    db = DBConnection({'type': 'oracle'})
    self.assertEqual(len(db.execute_query("SELECT *  FROM ALL_TAB_COLUMNS WHERE ROWNUM <= 1;")), 1)
    self.assertEqual(len(db.execute_query("SELECT *  FROM ALL_TAB_COLUMNS WHERE ROWNUM <= 1")), 1)
    self.assertEqual(len(db.execute_query("SELECT *  FROM ALL_TAB_COLUMNS LIMIT 1;")), 1)
    self.assertEqual(len(db.execute_query("SELECT *  FROM SE_PERSON WHERE ROWNUM <= 1;", {'SE_PERSON'})), 1)
    query = "SELECT *  FROM ALL_TAB_COLUMNS WHERE ROWNUM <= 1"
    for i in range(300):
      query = f"SELECT * FROM ({query})"
    self.assertEqual(len(db.execute_query(query)), 1)
    del db

  def test_spider(self):
    db = DBConnection({'type': 'sqlite3', 'db_path': '/data/spider/database/', 'db_id': 'student_transcripts_tracking'})
    self.assertEqual(len(db.execute_query("SELECT other_student_details FROM Students ORDER BY other_student_details DESC")), 15)


if __name__ == '__main__':
  unittest.main()
