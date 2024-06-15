from datetime import datetime
import json
import os
from enum import Enum

# INITIAL_PROMPT = "** TASK INSTRUCTION ** \nYour job is to write SQL queries that answer a user's question using the tables in the MIT Data Warehouse. \
# The MIT Data Warehouse is a central data source that combines data from various administrative systems at MIT, containing \
# information about students, faculty, and personnel. You can find more about the MIT data warehouse tables at \
# https://web.mit.edu/warehouse/metadata/tables/all_tables.html. \
# Reply with only the answer in SQL and include no linebreaks, newlines, escape characters or other commentary.\n\n"
# INITIAL_PROMPT = "Generate SQL given the question and tables to answer the question correctly.\n\n"
# INITIAL_PROMPT = "Your job is to write SQL queries that answer a user's question using the tables in the 'student_transcripts_tracking' database\
# in the Spider Dataset. \
# The 'student_transcripts_tracking' dataset likely comes from a specific data source used to create the Spider dataset. The Spider dataset \
# is a collection of question-answer pairs based on databases, and 'student_transcripts_tracking' is representing one source of these \
# databases on which question-answers pairs in the spider-dataset were being created.\n \
# Generate SQL for this database to answer the question correctly."
# INITIAL_PROMPT = "Generate SQL for a database to answer the question correctly.\n\n"
INITIAL_PROMPT = "Generate SQL given the question and tables to answer the question correctly.\n\n"
# INITIAL_PROMPT = "Generate SQL for the 'student_transcripts_tracking' database of the spider data to answer the following question correctly.\n\n"

BATCH_SIZE = 200


class EvaluationType(Enum):
  TABLE_NAME_EXTRACTION = 1  # tableretrieval
  VALUE_EXTRACTION = 2  # execution_accuracy


TIMEOUT_DURATION: int = 30  # Adjust this as needed


prefix = str(os.path.dirname(os.path.abspath(__file__)))
prefix = prefix.replace('/src/scripts', '')

SPIDER_DB_ID = 'student_transcripts_tracking'

MODELS = ['gpt-3.5-turbo-0125', 'gpt-3.5-turbo-16k', 'gpt-4', 'gpt-4-1106-preview']
model = MODELS[0]
EVALUATION_TYPE = EvaluationType.TABLE_NAME_EXTRACTION
PRIM_KEYS_ACTIVE = True
DATE_STRING = "2024-06-11"
if DATE_STRING is None:
  # Get the current date and time
  CURRENT_DATETIME = datetime.now()
  # Convert the current day to a string
  DATE_STRING = str(CURRENT_DATETIME.date())
K = 10
db_id = "fiben"
if db_id == "mit":
  db_type = "oracle"
else:
  db_type = "sqlite3"
DB_FILE = f'{db_id}.sqlite3'
if db_id == "spider":
  DB_FILE = 'student_transcripts_tracking/student_transcripts_tracking.sqlite'

with open(prefix + '/config.json') as f:
  CONFIG = json.load(f)
