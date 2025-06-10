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

#sample = random.sample(range(31), 10)
#print(sample)

from sql_metadata import Parser

from check import load_json_data
from generate import create_formated_output, create_tablestatements, create_examples, generate_combined_candidate
from retrieval import rank_sentences_more


with open("./data/bird/sample.json") as f:#sql_sample_50_ts_and_es_and_dec.json") as f:
    data = json.load(f)  # Should be a list of {"gold-sql": "...", ...}
#test = [0,1,6,9,18,20,24,25,28,29,35,38,40,41,45]
#test = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,24,25,26,27,28,29,35,38,40,41,45]
#print(len(test))

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


'''
# Execute a SELECT query
index = 0
for entry in data:
    sql = entry.get('gold-sql')
    db = entry.get('db_id')
    try:
        conn = sqlite3.connect(f"./data/bird/database/{db}/{db}.sqlite")
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        headers = [desc[0] for desc in cursor.description]
        #print(headers)
        #print(rows)

        # Combine headers and rows to compute max width for each column
        all_rows = [headers] + [list(map(str, row)) for row in rows]
        col_widths = [max(len(str(cell)) for cell in col) for col in zip(*all_rows)]

        # Format rows
        def format_row(row):
            return " | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths))

        # Create table string
        table_str = "\n".join([format_row(headers)] + ["-" * sum(col_widths) + "-" * (3 * (len(headers) - 1))] + [format_row(row) for row in rows])
        #print(table_str)

        # Close the connection
        conn.close()
        print('-')
    except Exception as e:
        print(index)
        print(e)
    index +=1

''' 
difficulty_counts = Counter()
failures = 0
difficulty_buckets = defaultdict(list)

sample = {"easy":[], "medium":random.sample(range(1020), 11), "hard":random.sample(range(283), 3), "extra":random.sample(range(104), 1)}
samples=[]

index = -1
for entry in data:
    index += 1
    #if index not in test:
    #  continue
    sql = entry.get("gold-sql") or entry.get("query")  # adapt key if needed
    #if entry.get("db_id") == "dw":
        #sql = sanitize_column_names(sql)
    try:
        parsed = sqlglot.parse_one(sql)
        features = extract_features(parsed)
        level = assign_difficulty(features)
        if level == "hard":
            print(index)
        difficulty_counts[level] += 1
        difficulty_buckets[level].append(entry)
        if len(difficulty_buckets[level]) in sample[level]:
            samples.append(entry)
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
sample = random.sample(range(1020), 11)
sample1 = random.sample(range(283), 3)
sample2 = random.sample(range(104), 1)
#with open("/home/skikk/Dokumente/uni/WS2324/MIT/enterprise-txt2sql/website/data/bird/sample.json", "w") as f:
#    json.dump(samples, f, indent=2)
