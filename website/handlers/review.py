import json
import re
import sqlite3
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from flask import render_template
from openai import OpenAI

from metrics import evaluate_nl_accuracy


# website/
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config.json"
DATA_DIR = BASE_DIR / "data"


def load_config():
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_runtime_config():
    config = load_config()

    user_id = config.get("user_id")
    api_key = config.get("api_key")
    database = config.get("database")

    if not user_id:
        raise ValueError("user_id not found in website/config.json")
    if not api_key:
        raise ValueError("api_key not found in website/config.json")
    if not database:
        raise ValueError("database not found in website/config.json")

    return {
        "user_id": user_id,
        "api_key": api_key,
        "database": database.lower(),
        "sql_file": DATA_DIR / "user" / f"queries_{user_id}.json",
    }


def get_db_path(db_id, database):
    """
    Current dataset layout:
    website/data/<dataset>/database/<db_id>/<db_id>.sqlite
    """
    dataset = database.lower()

    if dataset in {"spider", "bird", "beaver", "fiben", "sample"}:
        print(DATA_DIR / dataset / "database" / db_id / f"{db_id}.sqlite")
        return DATA_DIR / dataset / "database" / db_id / f"{db_id}.sqlite"
    
    raise ValueError(f"Unsupported DATABASE type: {dataset}")


def get_schema_text(db_id, database):
    db_path = get_db_path(db_id, database)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall() if row[0] != "sqlite_sequence"]

    schema_parts = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info('{table}')")
        cols = cursor.fetchall()
        col_defs = ", ".join([f"{col[1]} {col[2]}" for col in cols])
        schema_parts.append(f"Table {table}: {col_defs}")

    conn.close()
    return "\n".join(schema_parts)


def build_back_translation_prompt(question, schema_text, database):
    dataset = database.lower()
    dialect = "SQLite" if dataset in {"spider", "bird", "beaver", "fiben"} else "SQL"

    return f"""
You are a text-to-SQL system.

Dataset type: {dataset}
SQL dialect: {dialect}

Database schema:
{schema_text}

Task:
Write one executable SQL query for the following natural language question.

Question:
{question}

Rules:
- Return only SQL.
- Do not include markdown code fences.
- Do not explain anything.
- Use only tables and columns from the schema.
- Produce a single SELECT statement.
""".strip()


def call_llm_for_sql(prompt, api_key, model="gpt-4.1-mini"):
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You translate natural language questions into SQL queries."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def clean_sql(llm_output):
    sql = llm_output.strip()
    sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"^```\s*", "", sql)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip()


def execute_sql(db_id, sql, database):
    db_path = get_db_path(db_id, database)
    conn = sqlite3.connect(str(db_path))

    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        return {
            "ok": True,
            "rows": rows,
            "error": None
        }
    except Exception as e:
        return {
            "ok": False,
            "rows": None,
            "error": str(e)
        }
    finally:
        conn.close()


def normalize_value(v):
    if isinstance(v, float):
        return round(v, 6)
    return v


def normalize_rows(rows):
    return [tuple(normalize_value(x) for x in row) for row in rows]


def compare_execution_results(gold_rows, pred_rows):
    """
    Returns a score in [0, 1]
    1.0 = exact multiset match
    otherwise = row-overlap F1
    """
    gold_norm = normalize_rows(gold_rows)
    pred_norm = normalize_rows(pred_rows)

    gold_counter = Counter(gold_norm)
    pred_counter = Counter(pred_norm)

    if gold_counter == pred_counter:
        return 1.0

    intersection = sum((gold_counter & pred_counter).values())
    gold_total = sum(gold_counter.values())
    pred_total = sum(pred_counter.values())

    if gold_total == 0 and pred_total == 0:
        return 1.0
    if gold_total == 0 or pred_total == 0:
        return 0.0

    precision = intersection / pred_total
    recall = intersection / gold_total

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_back_translation_score(question, gold_sql, db_id, database, api_key):
    """
    NL question -> LLM SQL -> execute both -> compare outputs
    """
    if not question or not gold_sql:
        return 0.0, None, "missing question or gold_sql"

    try:
        schema_text = get_schema_text(db_id, database)
        prompt = build_back_translation_prompt(question, schema_text, database)
        print("HELLOOOOO")
        print(prompt)
        candidate_sql = clean_sql(call_llm_for_sql(prompt, api_key=api_key))

        # safety: only allow SELECT
        if not candidate_sql.strip().lower().startswith("select"):
            return 0.0, candidate_sql, "candidate_sql_not_select"

        gold_exec = execute_sql(db_id, gold_sql, database)
        pred_exec = execute_sql(db_id, candidate_sql, database)

        if not gold_exec["ok"]:
            return 0.0, candidate_sql, f"gold_sql_failed: {gold_exec['error']}"
        if not pred_exec["ok"]:
            return 0.0, candidate_sql, f"candidate_sql_failed: {pred_exec['error']}"

        score = compare_execution_results(gold_exec["rows"], pred_exec["rows"])
        return score, candidate_sql, None

    except Exception as e:
        print("Hi")
        print(str(e))
        return 0.0, None, str(e)


def _review():
    runtime = get_runtime_config()
    sql_file = runtime["sql_file"]
    database = runtime["database"]
    api_key = runtime["api_key"]

    if not sql_file.exists():
        raise FileNotFoundError(f"User query file not found: {sql_file}")

    logdata = pd.read_json(sql_file)
    logdata = logdata[logdata["question"] != ""].copy()

    generated_list = logdata["question"]
    reference_list = logdata["gold-question"]

    if len(reference_list) != len(generated_list):
        raise ValueError("Both lists must have the same length.")

    results = {
        "BLEU": [],
        "ROUGE": [],
        "BERTScore": [],
        "BACK_TRANSLATION": []
    }

    bt_sql_list = []
    bt_error_list = []

    for _, row in logdata.iterrows():
        ref = row["gold-question"]
        gen = row["question"]

        if not ref or not gen:
            scores = {"BLEU": 0.0, "ROUGE": 0.0, "BERTScore": 0.0}
            bt_score = 0.0
            bt_sql = None
            bt_error = "empty question"
        else:
            scores = evaluate_nl_accuracy(ref, gen)

            bt_score, bt_sql, bt_error = compute_back_translation_score(
                question=gen,
                gold_sql=row["gold-sql"],
                db_id=row["db_id"],
                database=database,
                api_key=api_key
            )

        results["BLEU"].append(float(scores["BLEU"]))
        results["ROUGE"].append(float(scores["ROUGE"]))
        results["BERTScore"].append(float(scores["BERTScore"]))
        results["BACK_TRANSLATION"].append(float(bt_score))

        bt_sql_list.append(bt_sql)
        bt_error_list.append(bt_error)

    logdata["back_translation_sql"] = bt_sql_list
    logdata["back_translation_error"] = bt_error_list
    logdata["back_translation_score"] = results["BACK_TRANSLATION"]

    result = {
        "adjusted": int(logdata["adjusted"].eq(True).sum()),
        "annotated": len(generated_list),
        "bleu": round(np.mean(results["BLEU"]) * 100, 2),
        "rouge": round(np.mean(results["ROUGE"]) * 100, 2),
        "bert": round(np.mean(results["BERTScore"]) * 100, 2),
        "back_translation": round(np.mean(results["BACK_TRANSLATION"]) * 100, 2)
    }

    print(result)
    return render_template("review.html", annotation=result)