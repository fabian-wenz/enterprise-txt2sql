import unittest

import os
import sys

sys.path.append(os.path.abspath('./'))
from src.scripts.prompt import *

table_schema = " ** TOP 1 MOST RELEVANT TABLES ** \n\
CREATE TABLE ACADEMIC_TERMS(\n\
ACADEMIC_TERMS_KEY VARCHAR2,\n\
TERM_CODE VARCHAR2,\n\
TERM_DESCRIPTION VARCHAR2,\n\
TERM_SELECTOR VARCHAR2,\n\
TERM_START_DATE DATE,\n\
TERM_END_DATE DATE,\n\
ACADEMIC_YEAR VARCHAR2,\n\
ACADEMIC_YEAR_DESC VARCHAR2,\n\
IS_CURRENT_TERM VARCHAR2,\n\
IS_REGULAR_TERM VARCHAR2,\n\
TERM_STATUS_INDICATOR VARCHAR2,\n\
TERM_STATUS VARCHAR2,\n\
FINANCIAL_AID_YEAR VARCHAR2,\n\
DEGREE_YEAR VARCHAR2,\n\
LAST_DAY_OF_FINAL_EXAM DATE,\n\
PRE_REGISTRATION_START_DAY DATE,\n\
REGISTRATION_DAY DATE,\n\
FIRST_DAY_OF_CLASSES DATE,\n\
LAST_DAY_OF_CLASSES DATE,\n\
ADD_DATE DATE,\n\
DROP_DATE DATE,\n\
GRADUATE_AWARD_START_DATE DATE,\n\
GRADUATE_AWARD_END_DATE DATE,\n\
WAREHOUSE_LOAD_DATE DATE)\n\n"
example = " ** 1-SHOT PSEUDO_EXAMPLE ** \n\
 ** PSEUDO EXAMPLE NR. 1 ** \n\
CREATE TABLE SE_PERSON(\n\
MIT_ID VARCHAR2,\n\
KRB_NAME VARCHAR2,\n\
FULL_NAME VARCHAR2,\n\
PAYROLL_RANK VARCHAR2,\n\
POSITION_TITLE VARCHAR2,\n\
IS_ACTIVE CHAR,\n\
OFFICE_LOCATION VARCHAR2,\n\
ORGANIZATION VARCHAR2,\n\
FIRST_NAME VARCHAR2,\n\
LAST_NAME VARCHAR2,\n\
MIDDLE_NAME VARCHAR2,\n\
EMPLOYEE_TYPE VARCHAR2)\n\
Question: When? \n\
Answer: SELECT * FROM SE_PERSON; \n\n"


class TestPrompt(unittest.TestCase):
  def test_empty_prompt(self):
    prompt = Prompt([], [])
    self.assertEqual(prompt.get_prompt('', set(), []), INITIAL_PROMPT + "Question: ")

  def test_simple_prompt(self):
    prompt = Prompt(['When?', 'Where?'], ['SELECT * FROM SE_PERSON;', 'SELECT * FROM SE_PERSON, ACADEMIC_TERMS;'])
    self.assertEqual(INITIAL_PROMPT + "Question: ", prompt.get_prompt('', set(), []))
    self.assertEqual(INITIAL_PROMPT + table_schema + "Question: ", prompt.get_prompt('', {'ACADEMIC_TERMS'}, []))
    self.assertEqual(INITIAL_PROMPT + example + "Question: ", prompt.get_prompt('', set(), [0]))
    self.assertEqual(INITIAL_PROMPT + example + table_schema + "Question: ",
                     prompt.get_prompt('', {'ACADEMIC_TERMS'}, [0]))
    print(prompt.get_prompt('', {'ACADEMIC_TERMS'}, [0]))


if __name__ == '__main__':
  unittest.main()
