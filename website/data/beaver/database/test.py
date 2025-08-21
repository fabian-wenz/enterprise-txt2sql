import json
import random
import sqlite3
import pandas as pd
import glob
import os
from collections import defaultdict, Counter
from openai import OpenAI
import sqlglot
import re

from sql_metadata import Parser

from check import load_json_data
from generate import create_formated_output, create_tablestatements, create_examples, generate_combined_candidate
from retrieval import rank_sentences_more

'''
# Parameters
csv_folder = "./dw/csv/"
sqlite_db_path = "./dw/dw.sqlite"

# Connect to SQLite database
conn = sqlite3.connect(sqlite_db_path)

# Loop through all CSV files in the folder
for csv_file in glob.glob(os.path.join(csv_folder, "*.csv")):
    table_name = os.path.splitext(os.path.basename(csv_file))[0]  # Use filename as table name
    print(f"Inserting {csv_file} into table '{table_name}'")

    try:
        df = pd.read_csv(csv_file)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()  # Flush changes after each file
    except Exception as e:
        print(f"Failed to insert {csv_file}: {e}")

# Close the connection
conn.close()
print("All CSV files inserted successfully.")
'''


'''
conn = sqlite3.connect("./dw/dw.sqlite")
cursor = conn.cursor()

# Execute a SELECT query
try:
    cursor.execute("SELECT sd.DEPARTMENT_NAME, sad.DEPARTMENT_PHONE_NUMBER, COUNT(DISTINCT msd.EMAIL_ADDRESS) AS Num_Students, MAX(LENGTH(msd.FULL_NAME)) AS Longest_Student_Name_Length FROM MIT_STUDENT_DIRECTORY msd JOIN SIS_DEPARTMENT sd ON msd.DEPARTMENT = sd.DEPARTMENT_CODE JOIN SIS_ADMIN_DEPARTMENT sad ON sd.DEPARTMENT_CODE = sad.SIS_ADMIN_DEPARTMENT_CODE GROUP BY sd.DEPARTMENT_NAME, sad.DEPARTMENT_PHONE_NUMBER;")
    rows = cursor.fetchall()
    headers = [desc[0] for desc in cursor.description]
    print(headers)
    print(rows)

    # Combine headers and rows to compute max width for each column
    all_rows = [headers] + [list(map(str, row)) for row in rows]
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*all_rows)]

    # Format rows
    def format_row(row):
        return " | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths))

    # Create table string
    table_str = "\n".join([format_row(headers)] + ["-" * sum(col_widths) + "-" * (3 * (len(headers) - 1))] + [format_row(row) for row in rows])
    print(table_str)

    # Close the connection
    conn.close()
except Exception as e:
    print(e)

'''



















'''
def sanitize_column_names(sql):
    # Replace problematic names with underscored equivalents
    return re.sub(r"`([^`]+?) \(([^`]+?)\)`", lambda m: f"`{m.group(1).replace(' ', '_')}_{m.group(2).replace('-', '')}`", sql)

def assign_difficulty(features):
    if features["has_subquery"]:
        return "extra"
    elif features["num_joins"] >= 2 or features["has_group_by"]:
        return "hard"
    elif features["num_joins"] == 1 or features["num_conditions"] > 2:
        return "medium"
    else:
        return "easy"

def extract_features(parsed_query):
    return {
        "num_joins": len(list(parsed.find_all(sqlglot.exp.Join))),
        "has_group_by": parsed.find(sqlglot.exp.Group) is not None,
        "has_order_by": parsed.find(sqlglot.exp.Order) is not None,
        "has_having": parsed.find(sqlglot.exp.Having) is not None,
        "has_subquery": any(isinstance(n, sqlglot.exp.Subquery) for n in parsed.walk()),
        "num_conditions": len(list(parsed.find_all(sqlglot.exp.Condition))),
    }


# === Load your JSON file ===
with open("../../beaver/queries_copy.json") as f:
    data = json.load(f)  # Should be a list of {"gold-sql": "...", ...}

# === Analyze all queries ===
difficulty_counts = Counter()
failures = 0
difficulty_buckets = defaultdict(list)

for entry in data:

    sql = entry.get("gold-sql") or entry.get("query")  # adapt key if needed
    #if entry.get("db_id") == "dw":
        #sql = sanitize_column_names(sql)
    try:
        parsed = sqlglot.parse_one(sql)
        features = extract_features(parsed)
        level = assign_difficulty(features)
        difficulty_counts[level] += 1
        difficulty_buckets[level].append(entry)
    except Exception as e:
        failures += 1
        print(f"Failed to parse SQL: {sql}\nError: {e}")

# === Output summary ===
print("SQL Difficulty Breakdown:")
for level in ["easy", "medium", "hard", "extra"]:
    print(f"{level}: {difficulty_counts[level]}")

print(f"Failed parses: {failures}")


# Total available per class
counts = {k: len(v) for k, v in difficulty_buckets.items()}
print(counts)
total = sum(counts.values())

# Sample size
sample_size = 50

# Stratified sample allocation
sample_counts = {
    k: max(1, round(sample_size * (v / total))) for k, v in counts.items()
}
print(sample_counts)

# Fix rounding error
adjustment = sample_size - sum(sample_counts.values())
if adjustment != 0:
    # Adjust largest class to fix
    max_k = max(sample_counts, key=sample_counts.get)
    sample_counts[max_k] += adjustment

# Draw samples
final_sample = []
for k, n in sample_counts.items():
    final_sample.extend(random.sample(difficulty_buckets[k], min(n, len(difficulty_buckets[k]))))

# Output or save sample
with open("/home/skikk/Dokumente/uni/WS2324/MIT/enterprise-txt2sql/website/data/beaver/sql_sample_50.json", "w") as f:
    json.dump(final_sample, f, indent=2)
'''
SCHEMA_FILE = "../tables.json"

def clean(option):
    words = option.split()
    if not words:
        return option
    first = words[0]
    if len(first) < 5 and (re.search(r'[\d\.\-]', first) or not first.isalpha()):
        return " ".join(words[1:])
    return option

def extract_cte_titles(sql):
    try:
        parsed = sqlglot.parse_one(sql)
        with_expr = parsed.args.get("with")
        if not with_expr:
            return []
        return [cte.alias for cte in with_expr.expressions]
    except Exception as e:
        print(f"Failed to parse SQL: {e}")
        return []

api_key="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"
prompt_text= "\nEach option should:\n1. **Clearly describe the purpose of the SQL query**, emphasizing what the query aims to achieve.\n2. **Refer to each column in the output**, paying particular attention to their **order**. If necessary, explain what each column represents.\n3. **Explain any calculations** (e.g., sums, averages) and how they are derived.\n4. **Summarize the final result of the query**, but **avoid referencing SQL constructs** like CTEs, joins, or unions.\n5. Use **diverse phrasings and question structures** to avoid redundancy and enhance clarity. Each option should be distinct but accurate to the SQL query."
#prompt= f"Your task is to generate 4 different natural language options for the given SQL query. \n\nThe goal is for the user to later select the most appropriate natural language question from these options. \n\n {prompt_text} \n\n### Now, Perform the Task for the Given SQL Query:\n\n"

#prompt_basic = f"Your task is to generate 4 different natural language options for the given SQL query. \n\nThe goal is for the user to later select the most appropriate natural language question from these options. \n\n{prompt_text}\n#### Example SQL Query & gold natural language description:\n\n{examples}\n\n#### Relevant Table Schema for the upcoming task:\n\n{schema}\n\n### Now, Perform the Task for the Given SQL Query:\n\n"
#prompt = f"Your task is to generate 4 different natural language options for the given SQL query. \n\nThe goal is for the user to later select the most appropriate natural language question from these options. \n\n{prompt_text}\n#### Example SQL Query & gold natural language description:\n\n{examples}\n\n#### Relevant Table Schema for the upcoming task:\n\n{schema}\n\n### Now, Perform the Task for the Given SQL Query:\n\n",

model = "gpt-3.5-turbo"
def get_sqlite_logical_plan(sql_query: str) -> str:
    explain_query = f"EXPLAIN QUERY PLAN {sql_query}"
    cursor.execute(explain_query)
    rows = cursor.fetchall()
    plan_lines = [" ".join(str(col) for col in row) for row in rows]
    return "\n".join(plan_lines)

with open("../../beaver/sql_sample_50_ts_and_es_and_dec.json") as f:
    data = json.load(f)  # Should be a list of {"gold-sql": "...", ...}

with open("../../beaver/sql_sample_50.json") as f:
    data_ = json.load(f)  # Should be a list of {"gold-sql": "...", ...}

for i in range(len(data)):
    if "sql_decomposition" in data[i].keys():
        data[i]["sql_embedding"] = data_[i]["sql_embedding"]

with open("../../beaver/queries_copy.json") as f:
    other_data = json.load(f)  # Should be a list of {"gold-sql": "...", ...}

directory_files = set(os.listdir("../../beaver/schema"))

annotated = []
annotated_sqls = [data_entry.get("oracle_sql") for data_entry in data]
for other_entries in other_data:
    if other_entries.get("oracle_sql") not in annotated_sqls:
        annotated.append(other_entries)

conn = sqlite3.connect("./dw/dw.sqlite")
cursor = conn.cursor()
for entry in data:
    if entry.get("sql_decomposition"):
        #entries = entries.get("sql_decomposition")
        #for entry in entries:
        sql = entry.get("oracle_sql") or entry.get("gold-sql")

        sql_embedding = [entry.get("sql_embedding") for entry in annotated if isinstance(entry, dict)]
        gold_sql =[entry.get("oracle_sql")or entry.get("gold-sql") for entry in annotated if isinstance(entry, dict)]
        gold_question = [entry.get("gold-question") for entry in annotated if isinstance(entry, dict)]
        most_relevant_examples_ = rank_sentences_more(entry.get('sql_embedding'),gold_sql, sql_embedding,gold_question)
        most_relevant_examples = [{'sql':x[0],'question':x[1]} for x in most_relevant_examples_]


        examples = create_examples(most_relevant_examples)[:5]
        try:
            tables = Parser(sql).tables
            tables = list(directory_files.intersection(set(["DW-" + table.upper() + ".csv" for table in tables])))
            tables = [table[3:-4] for table in tables]
        except Exception as e:
            print(e)
            tables = []
        schema = create_tablestatements(tables, '../schema/', 'dw')

        # Execute a SELECT query
        try:
            try:
                cursor.execute(sql)
                rows = cursor.fetchall()
                headers = [desc[0] for desc in cursor.description]
                explain_query = f"EXPLAIN QUERY PLAN {sql}"
                cursor.execute(explain_query)
                plan_lines = [" ".join(str(col) for col in row) for row in cursor.fetchall()]
            except:
                rows = []
                headers = []

            '''prompt = f"Your task is to generate 4 different natural language options for the given SQL query. \n\nThe goal is for the user to later select the most appropriate natural language question from these options. \n\n{prompt_text}\n#### Example SQL Query & gold natural language description:\n\n{examples}\n\n#### Relevant Table Schema for the upcoming task:\n\n{schema}\n\n### Now, Perform the Task for the Given SQL Query:\n\n",

            client = OpenAI(api_key=api_key)
            print(str(prompt))
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": str(prompt)
                    },
                    {
                        "role": "user",
                        "content": sql + ("\n### the Output from the above SQL:\n" + create_formated_output(rows, headers) if rows else "")
                    }
                ]
            )

            res = response.choices[0].message.content
            options = [line.strip() for line in res.splitlines() if line.strip() and "OPTION" not in line.upper() and ":" not in line]
            options = [clean(option) for option in options]
            entry['question_ts_and_es'] = options[0]
            entry['options'] = options'''
            titles = extract_cte_titles(entry['sql_in_cte'])
            annotations = {}
            for i in range(len(titles)):
                annotations[titles[i]] = entry['sql_decomposition'][i]['question']
            annotations['main'] = entry['sql_decomposition'][len(titles)]['question']
            #entry['options'] = generate_combined_candidate(model, api_key, entry['sql_in_cte'],annotations, examples[:1], prompt_text, rows, headers)
            entry['options'] = generate_combined_candidate(model, api_key, entry['sql_in_cte'],annotations, plan_lines, prompt_text, rows, headers)
            entry['question_dec'] = entry['options'][0]
        except Exception as e:
            print(e)

    # Presumably, you're adding something to final_sample here? If not, this needs clarification.

with open("/home/skikk/Dokumente/uni/WS2324/MIT/enterprise-txt2sql/website/data/beaver/sql_sample_50_ts_and_es_and_decomposed.json", "w") as f:
    json.dump(data, f, indent=2)
