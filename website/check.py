import csv
import os
import json
import re
import sqlite3
import pandas as pd
import sqlglot
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import sqlparse


def validate_json(json_path, errors):
  """Check if JSON file has a list of dictionaries with 'sql' and 'question' keys."""
  if not os.path.exists(json_path):
    errors.append(f"JSON file not found: {json_path}")
    return False

  try:
    with open(json_path, 'r', encoding='utf-8') as file:
      data = json.load(file)

    if not isinstance(data, list):
      errors.append("JSON file does not contain a list.")
      return False

    for entry in data:
      if not isinstance(entry, dict) or 'sql' not in entry or 'question' not in entry:
        errors.append("JSON file contains invalid entries (missing 'sql' or 'question').")
        return False
    for entry in data:
      entry["gold-sql"] = entry.pop("sql")
      entry["sql"] = ""
      entry["gold-question"] = entry.pop("question")
      entry["question"] = ""
      entry["adjusted"] = ""
      entry["comment"] = ""
      if 'db_id' not in entry.keys():
        entry["db_id"] = ""
      entry["options"] = ""
    if data:
      with open(json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


    return True

  except json.JSONDecodeError:
    errors.append("JSON file is not properly formatted.")
    return False


def validate_sqlite(db_path, errors):
  """Check if the SQLite file exists and is a valid database."""
  if not os.path.exists(db_path):
    errors.append(f"SQLite database file not found: {db_path}")
    return False

  try:
    # Try connecting to the database
    conn = sqlite3.connect(db_path)
    conn.close()
    return True

  except sqlite3.Error:
    errors.append("SQLite file is not a valid database.")
    return False


def validate_schema_csv(csv_path, errors):
  """Check if CSV file contains required columns: table, columnname, data-type, description."""
  required_columns = {"table", "columnname", "data-type", "description"}

  if not os.path.exists(csv_path):
    errors.append(f"Schema CSV file not found: {csv_path}")
    return False

  try:
    df = pd.read_csv(csv_path)
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
      errors.append(f"Schema CSV is missing columns: {missing_columns}")
      return False

    return True

  except Exception as e:
    errors.append(f"Error reading CSV file: {e}")
    return False


def check_files(json_path, db_path, csv_path):
  """Run all validation checks and collect error messages."""
  errors = []

  print("\nChecking JSON file...")
  json_valid = validate_json(json_path, errors)

  print("\nChecking SQLite file...")
  sqlite_valid = validate_sqlite(db_path, errors)

  print("\nChecking Schema CSV file...")
  schema_valid = validate_schema_csv(csv_path, errors)

  return (json_valid and sqlite_valid and schema_valid, errors)
  # if json_valid and sqlite_valid and schema_valid:
  #    print("\n✅ All files are valid!")
  # else:
  #    print("\n❌ Some files have issues. Please check the messages above.")


def embedding_json(json_path):
  # Encode reference sentence and detach
  model = SentenceTransformer("all-MiniLM-L6-v2")
  data = pd.read_json(json_path)
  data['sql_embedding'] = list(model.encode(data['sql'].to_list(), convert_to_tensor=True).detach().cpu().numpy())

  data['question_embedding'] = list(
    model.encode(data['gold-question'].to_list(), convert_to_tensor=True).detach().cpu().numpy())
  data.to_json(json_path, orient="records")


def embedding_csv(csv_path):
  # Encode reference sentence and detach
  model = SentenceTransformer("all-MiniLM-L6-v2")
  data = pd.read_json(csv_path)
  data['schema_embedding'] = list(model.encode(data['schema'].to_list(), convert_to_tensor=True).detach().cpu().numpy())

  data.to_json(csv_path[:-4] + '.json', orient="records")

def check_db_id(json_file):
  logdata = pd.read_json(json_file)
  # Fetch the specific annotation based on the ID provided in the URL
  for i in range(len(logdata)):
    if i in[10,11, 12]:
      continue
    try:
      conn = sqlite3.connect("./data/fiben/database/fiben.sqlite")

      #conn = sqlite3.connect("./data/fiben/database/"+logdata['db_id'][i] + "/"+logdata['db_id'][i] + ".sqlite")
      cursor = conn.cursor()

      # Execute a SELECT query
      cursor.execute(logdata['gold-sql'][i].replace("FIBEN.", ""))
      print(cursor.fetchall()[:10])
      print([desc[0] for desc in cursor.description])
      print('--------------------------------')
      print(i)
      print(logdata['gold-sql'][i])
      print('--------------------------------')
    except Exception as e:
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print(e)
      print(logdata['gold-sql'][i])
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    conn.close()
def load_json_data(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data

def save_json_data(file_name, data):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def convert_schema(json_file):
  df = load_json_data(json_file)
  for i in range(len(df)):
    j, k = 0, 0
    while(k < (len(df[i]['table_names_original']))):
      schema_file = df[i]['db_id'].upper() + '-'
      data = []
      schema_file += df[i]['table_names_original'][k].upper()
      foreign_keys = {}
      for c in df[i]['foreign_keys']:
        foreign_keys[c[0]] = c[1]
      while( j < (len(df[i]['column_names_original']))):
        if k == df[i]['column_names_original'][j][0]:
          prim = "PRIMARY KEY" if j in df[i]['primary_keys'] else ""
          frgn = ""
          if j in foreign_keys.keys():
            frgn_table = df[i]['table_names_original'][df[i]['column_names_original'][foreign_keys[j]][0]]
            frgn = "FOREIGN KEY "+ df[i]['db_id'].upper() + '-' + frgn_table.upper() + " ("+ df[i]['column_names_original'][foreign_keys[j]][1]+")"
          entry = {"COLUMN_NAME":df[i]['column_names_original'][j][1],"DATA_TYPE":df[i]['column_types'][j],"PKEY":prim,"FKEY":frgn}
          data.append(entry)
        elif k < df[i]['column_names_original'][j][0]:
          break
        j+=1;
      k+=1;
      print(schema_file)
      #df = pd.DataFrame(data)
      #df.to_csv('./data/bird/schema/'+schema_file + '.csv', index=False)
      with open('./data/bird/schema/'+schema_file + '.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=["COLUMN_NAME", "DATA_TYPE", "PKEY", "FKEY"])
        writer.writeheader()  # Write the header row
        writer.writerows(data)  # Write the data rows


def convert_schema_beaver(json_file):
  df = load_json_data(json_file)
  schema = []
  for i in df:
    j= 0
    schema_file = df[i]['db_id'].upper() + '-'
    data = []
    schema_file += df[i]['table_name_original'].upper()
    schema.append({'schema': schema_file + ',' + ','.join(df[i]['column_names_original'])})
    foreign_keys = {}
    if df[i]['db_id'] != 'dw':
      for c in df[i]['foreign_key']:
        referenced_table_name = c['referenced_table_name'].split('#')
        foreign_keys[c['column_name']] = "FOREIGN KEY {} ({})".format('-'.join([referenced_table_name[0], referenced_table_name[2]]), c['referenced_column_name'])
    while( j < (len(df[i]['column_names_original']))):
      frgn, prim = "", ""
      if df[i]['db_id'] != 'dw':
        prim = "PRIMARY KEY" if df[i]['column_names_original'][j] in df[i]['primary_key'] else ""
        if df[i]['column_names_original'][j] in foreign_keys.keys():
          frgn = foreign_keys[df[i]['column_names_original'][j]]
      entry = {"COLUMN_NAME":df[i]['column_names_original'][j],"DATA_TYPE":df[i]['column_types'][j],"PKEY":prim,"FKEY":frgn}
      data.append(entry)
      j+=1;
    print(schema_file)
    #df = pd.DataFrame(data)
    #df.to_csv('./data/bird/schema/'+schema_file + '.csv', index=False)
    with open('./data/beaver/schema/'+schema_file + '.csv', 'w') as f:
      writer = csv.DictWriter(f, fieldnames=["COLUMN_NAME", "DATA_TYPE", "PKEY", "FKEY"])
      writer.writeheader()  # Write the header row
      writer.writerows(data)  # Write the data rows
  save_json_data('./data/beaver/tables.json', schema)

def spider_short(json_file):
  data = load_json_data(json_file)
  i=0
  data_short = []
  while i+1< int(len(data)):
    if data[i]['gold-sql'].upper() == data[i+1]['gold-sql'].upper() or data[i]['gold-sql'].upper() in data[i+1]['gold-sql'].upper() or data[i+1]['gold-sql'].upper() in data[i]['gold-sql'].upper():
      data_short.append(data[i])
      i=i+2
    else:
      print(i)
      data_short.append(data[i])
      print(data[i]['gold-sql'])
      print(data[i+1]['gold-sql'])
      i=i+1
  save_json_data(json_file, data_short)

def decompose_statements(json_file):
  logdata = load_json_data(json_file)
  # Fetch the specific annotation based on the ID provided in the URL

  api_key = "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"
  client = OpenAI(api_key=api_key)
  for i in range(len(logdata)):
    nested_sql = logdata[i]['gold-sql'].replace("FIBEN.", "")
    if len(re.findall(r"\bSELECT\b", nested_sql, flags=re.IGNORECASE)) >=2:
      prompt = f"""
        I have the following SQL query that contains nested potentially correlated subqueries. Please rewrite it using WITH clauses (Common Table Expressions) instead of nested SELECT statements. 
        Make sure the logic remains equivalent. Please return only the rewritten SQL using CTEs.
        
        Here is the query:
        
        {nested_sql}"""
      response = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
              {"role": "system", "content": "You are an expert SQL refactoring assistant."},
              {"role": "user", "content": prompt}
          ],
          temperature=0,
      )
      logdata[i]['sql_in_cte']=response.choices[0].message.content
      print(logdata[i]['sql_in_cte'])

  save_json_data(json_file[:-4] + '2.json', logdata)

def reiterate_decompose(json_file):
  logdata = load_json_data(json_file)
  # Fetch the specific annotation based on the ID provided in the URL

  api_key = "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"
  client = OpenAI(api_key=api_key)
  conn = sqlite3.connect("./data/beaver/database/dw/dw.sqlite")
  cursor = conn.cursor()
  for i in range(len(logdata)):
    nested_sql = logdata[i]['gold-sql'].replace("FIBEN.", "")
    if len(re.findall(r"\bSELECT\b", nested_sql, flags=re.IGNORECASE)) >=2:
      sql_cte = logdata[i]['sql_in_cte']
      try:
        cursor.execute(sql_cte.replace("FIBEN.", ""))
      except Exception as e1:
        parsed = sqlglot.parse_one(sql_cte)

        # Get CTEs
        with_clause = parsed.args.get("with")
        error_in_cte = False
        for cte in with_clause.expressions:
          cte_name = cte.alias_or_name
          cte_sql = cte.this.sql(pretty=False)
          try:
             cursor.execute(cte_sql)
          except Exception as e2:
            print(e2)
            error_in_cte = True
            break

        print('--------------------------------')
        print(i)
        error = str(e1)

        start = sql_cte.rfind(cte_sql)
        end = start + len(cte_sql)
        error_part = sql_cte[end:]
        if error_in_cte:
          error_part = cte_name
        print(error)
        sql = logdata[i]['gold-sql'].replace("FIBEN.", "")
        prompt = f"""
          The following original SQL query works in SQLite:

          {sql}
          
          GPT rewrote it using CTEs, but the following rewritten version fails:
          
          {cte_sql}
          
          This part ``` {error_part} ``` fails with this SQLite error:
          
          {error}
          
          Please fix only that mistake and return the whole corrected query in CTE style, valid for SQLite. 
          Keep the original logic and structure. Before outputting the final query go through the result step-by-step, table-by-table and make sure the error does not exist anymore. 
          Do not include explanations or code blocks. Return only the corrected SQL query and nothing more. 
          """
        print(prompt)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert SQL refactoring assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        raw_response=response.choices[0].message.content
        if raw_response.startswith("```sql") or raw_response.startswith("```"):
            cleaned_sql = re.sub(r"^```sql\s*|```$", "", raw_response.strip(), flags=re.MULTILINE).strip()
        else:
            cleaned_sql = raw_response
        logdata[i]['sql_in_cte'] = cleaned_sql
        print(logdata[i]['sql_in_cte'])
        print('--------------------------------')

  save_json_data(json_file[:-4] + '3.json', logdata)

def extract_decomposition(json_file):
  logdata = load_json_data(json_file)
  model = SentenceTransformer("all-MiniLM-L6-v2")
  # Fetch the specific annotation based on the ID provided in the URL
  for i in range(len(logdata)):
    print(i)
    nested_sql = logdata[i]['gold-sql'].replace("FIBEN.", "")
    if len(re.findall(r"\bSELECT\b", nested_sql, flags=re.IGNORECASE)) >=2:
      try:
        print(logdata[i]['sql_in_cte'])
        sql_cte = logdata[i]['sql_in_cte']
        parsed = sqlglot.parse_one(sql_cte)

        # Get CTEs
        with_clause = parsed.args.get("with")
        ctes = {}
        for cte in with_clause.expressions:
          cte_name = cte.alias_or_name
          cte_sql = cte.this.sql(pretty=False)
          ctes[cte_name] = cte_sql

        start = sql_cte.rfind(cte_sql)
        end = start + len(cte_sql)
        temp_sql = sql_cte[end:]
        end += temp_sql.find('SELECT')
        ctes['main'] = sql_cte[end:]
        logdata[i]['sql_decomposition'] = []
        j=0
        for cte in ctes.keys():
          logdata[i]['sql_decomposition'].append({'question':"", "gold-sql":"", "title":"", "db_id":"","adjusted":False, "comment":"", "sql_embedding":[], "options":[]})
          logdata[i]['sql_decomposition'][j]['title'] = cte
          logdata[i]['sql_decomposition'][j]["gold-sql"] = ctes[cte]
          logdata[i]['sql_decomposition'][j]['db_id'] = logdata[i]['db_id']
          logdata[i]['sql_decomposition'][j]['sql_embedding'] = model.encode(logdata[i]['sql_decomposition'][j]["gold-sql"], convert_to_tensor=True).detach().cpu().numpy().tolist()
          j+=1
      except Exception as e:
        print(e)
        continue
  save_json_data(json_file[:-4] + '4', logdata)

def embedding_calc(json_file):
  logdata = load_json_data(json_file)
  model = SentenceTransformer("all-MiniLM-L6-v2")
  for entry in logdata:
    if not entry.get('sql_embedding'):
        entry['sql_embedding'] = model.encode(entry["gold-sql"], convert_to_tensor=True).detach().cpu().numpy().tolist()
    if not entry.get('sql_embedding'):
        entry['sql_embedding'] = model.encode(entry["gold-sql"], convert_to_tensor=True).detach().cpu().numpy().tolist()
    if entry.get('sql_decomposition'):
      for i in range(len(entry.get('sql_decomposition'))):
        entry['sql_decomposition'][i]['sql_embedding'] = model.encode(entry['sql_decomposition'][i]["gold-sql"], convert_to_tensor=True).detach().cpu().numpy().tolist()

  save_json_data(json_file[:-5] + '4.json', logdata)

if __name__ == '__main__':
  # Example file paths (replace with actual file paths)
  json_file = "./data/sample/queries.json"
  sqlite_file = "database.sqlite"
  csv_file = "./data/beaver/tables.csv"
  #embedding_csv(json_file)
  #convert_schema_beaver(json_file)
  #validate_json(json_file, [])
  #embedding_json(json_file)
  #spider_short(json_file)
  #




  #reiterate_decompose(json_file)
  #log_data = load_json_data(json_file)
  #for i in range(len(log_data)):
  #  if 'sql_in_cte' in log_data[i].keys():
  #    print('--------------------------------')
  #    print(i)
  #    print(sqlparse.format(log_data[i]['sql_in_cte'], reindent=True, keyword_case='upper'))
  #    print(sqlparse.format(log_data[i]['gold-sql'], reindent=True, keyword_case='upper'))
  #    print('--------------------------------')
  #decompose_statements(json_file)
  #reiterate_decompose(json_file)
  #extract_decomposition(json_file)
  embedding_calc(json_file)








  #validate_json(json_file, [])
  #embedding_json(json_file)
  #convert_schema(json_file)
  #

  # Run the checks
  # check_files(json_file, sqlite_file, csv_file)
