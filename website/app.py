import os
import sqlite3
import textwrap
import time

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, json, jsonify
import sqlparse
from generate import generate_candidate, generate_improved_prompt, generate_combined_candidate
from retrieval import rank_sentences, rank_sentences_more
import numpy as np
import ast
from sql_metadata import Parser
from metrics import evaluate_nl_accuracy


app = Flask(__name__)
with open("config.json", "r") as json_file:
    DATA = json.load(json_file)
    print(DATA)
DATABASE = DATA ['datasets'][1]
TASK = DATA['tasks'][0]
MODEL = DATA['models'][2]
REL_TABLES = []
REL_EXAMPLES = []
PROMPT = DATA['prompt']
PROMPT_TXT = DATA['prompt_text']
SQL_FILE = "./data/{DATABASE}/queries"
SCHEMA_FILE = "./data/{DATABASE}/tables"
SCHEMA_FOLDER = "./data/{DATABASE}/schema/"
OVERALL_TIME = ""
SINGLE_TIME = ""
USER_NAME = ""
#import time
#
#start = time.time()
## ... some code ...
#end = time.time()
#
#elapsed_seconds = end - start
#elapsed_milliseconds = (end - start) * 1000
#elapsed_minutes = (end - start) / 60
#
#print(f"Elapsed: {elapsed_seconds:.3f} sec")
#print(f"Elapsed: {elapsed_milliseconds:.1f} ms")
#print(f"Elapsed: {elapsed_minutes:.2f} min")
API_KEY = ""# "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" #"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"
def load_json_data(file_name):
    with open(file_name + '.json', 'r') as f:
        data = json.load(f)
    return data

def retrieve_filenames(folder_path):
    # Create an empty list to store the file names
    files = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a CSV file
        name, extension = os.path.splitext(filename)
        files.append(name)
    return files

def save_json_data(file_name, data):
    with open(file_name + '.json', 'w') as f:
        json.dump(data, f, indent=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/submit", methods=["POST"])
def submit():
    global API_KEY, USER_NAME, SQL_FILE  # Declare globals
    # Retrieve data from the form
    api_key = request.form.get("api_key")
    username = request.form.get("username")
    sql_file = request.files.get("sql_file")

    API_KEY = api_key
    USER_NAME = username

    # Save the SQL file locally (optional)
    if sql_file:
        sql_file.save(f"./{sql_file.filename}")
        if API_KEY != "":
            SQL_FILE = sql_file[:-4]
            #generate_candidates(API_KEY + '.json', SQL_FILE)


    # Process the API key, username, and SQL file as needed
    print(f"Username: {username}")
    print(f"SQL File: {sql_file.filename if sql_file else 'No file uploaded'}")

    # Redirect back to the homepage after submission
    return redirect(url_for("index"))
@app.route("/save_api_key_and_user_id", methods=["POST"])
def save_api_key_and_user_id():
    global API_KEY
    data = request.json
    api_key = data.get("api_key")

    if not api_key:
        return jsonify({"message": "API Key is missing!"}), 400

    API_KEY = api_key  # Store in memory (or a database)
    print("Received API Key:", API_KEY)

    return jsonify({"message": "API Key saved successfully!"})

@app.route('/decomposed_sql_annotation/<int:annotation_id>')
def decomposed_sql_annotation(annotation_id):

    global PROMPT, PROMPT_TXT,API_KEY, MODEL, SQL_FILE, DATABASE, REL_TABLES, REL_EXAMPLES
    logdata = load_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()))
    # Fetch the specific annotation based on the ID provided in the URL

    annotation = logdata[annotation_id - 1]
    data_length = len(logdata)
    num_annotated=0
    for i in range(len(logdata)):
        if logdata[i]['question'] != "":
            num_annotated+=1
    annotation['percentage'] = round(num_annotated / data_length *100,2)
    annotation['gold_sql'] = sqlparse.format(annotation['gold-sql'], reindent=True, keyword_case='upper').strip()
    annotations = annotation
    conn = sqlite3.connect("./data/"+DATABASE.lower()+"/database/"+annotation['db_id'] + "/"+annotation['db_id'] + ".sqlite")
    cursor = conn.cursor()

    # Execute a SELECT query
    try:
        cursor.execute(annotation['gold-sql'].replace("FIBEN.", ""))
        annotations['rows'] = cursor.fetchall()
        annotations['rows'] = annotation['rows'][:min(10, len(annotation['rows']))]
        annotations['column_names'] = [desc[0] for desc in cursor.description]

        # Close the connection
        conn.close()
    except Exception as e:
        print(e)
        annotations['rows'] =[[str(e)]]
        annotations['column_names'] = ["ERROR"]
    for i in range(len(annotations['sql_decomposition'])):
        annotations['sql_decomposition'][i]['gold_sql']= sqlparse.format(annotations['sql_decomposition'][i]['gold-sql'], reindent=True, keyword_case='upper').strip()
        annotations['sql_decomposition'][i]['options'] = generate_candidate(MODEL, API_KEY, PROMPT, PROMPT_TXT, annotations['sql_decomposition'][i]['gold-sql'],REL_TABLES[annotation_id][annotations['sql_decomposition'][i]['title']], REL_EXAMPLES[annotation_id][annotations['sql_decomposition'][i]['title']], annotation['db_id'], DATABASE, )
    logdata[annotation_id - 1] = annotations
    #save_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()), logdata)
    if "comment" not in annotation.keys():
        annotations['comment'] = ""
    return render_template('decomposed_sql_annotation.html', annotations=annotations, annotation_id=annotation_id, data_length=data_length)

@app.route('/sql_annotation/<int:annotation_id>')
def sql_annotation(annotation_id):

    global PROMPT, PROMPT_TXT,API_KEY, MODEL, SQL_FILE, DATABASE, REL_TABLES, REL_EXAMPLES

    logdata = load_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()))
    # Fetch the specific annotation based on the ID provided in the URL
    annotation = logdata[annotation_id - 1]
    data_length = len(logdata)
    num_annotated=0
    for i in range(len(logdata)):
        if logdata[i]['question'] != "":
            num_annotated+=1
    annotation_ = annotation
    annotation_['percentage'] = round(num_annotated / data_length *100,2)
    annotation_['gold_sql'] = sqlparse.format(annotation_['gold-sql'], reindent=True, keyword_case='upper')
    conn = sqlite3.connect("./data/"+DATABASE.lower()+"/database/"+annotation_['db_id'] + "/"+annotation_['db_id'] + ".sqlite")
    cursor = conn.cursor()

    # Execute a SELECT query
    try:
        cursor.execute(annotation_['gold-sql'].replace("FIBEN.", ""))
        annotation_['rows'] = cursor.fetchall()
        annotation_['rows'] = annotation_['rows'][:min(10, len(annotation['rows']))]
        annotation_['column_names'] = [desc[0] for desc in cursor.description]

        # Close the connection
        conn.close()
    except Exception as e:
        print(e)
        annotation_['rows'] =[[str(e)]]
        annotation_['column_names'] = ["ERROR"]
    annotation['options'] = generate_candidate(MODEL, API_KEY, PROMPT, PROMPT_TXT, annotation['gold-sql'],REL_TABLES[annotation_id], REL_EXAMPLES[annotation_id], annotation['db_id'], DATABASE, annotation_['rows'], annotation_['column_names'])
    logdata[annotation_id - 1] = annotation
    save_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()), logdata)
    if "comment" not in annotation.keys():
        annotation['comment'] = ""
    return render_template('sql_annotation.html', annotation=annotation, annotation_id=annotation_id, data_length=data_length)

@app.route("/task_selection")
def task_selection():
    #["Txt-2-SQL", "SQL-2-Txt", "Schema reconstruction", "NL question generation"],*/
    annotation = {'tasks': DATA["tasks"], 'models': DATA["models"], 'task':TASK, 'model': MODEL}
    print(annotation)
    return render_template("task_selection.html", annotation=annotation)


@app.route('/save_and_next/', methods=['POST'])
def save_and_next():
    global DATABASE, TASK, REL_TABLES, REL_EXAMPLES, SQL_FILE, MODEL, DATABASE, OVERALL_TIME, SINGLE_TIME
    type = request.args.get('type')
    if type == 'upload':
        DATABASE = request.form.get('selected_option')
        logdata = load_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()))
        data_length = len(logdata)
        REL_TABLES = ['']*data_length
        REL_EXAMPLES = ['']*data_length
        return redirect(url_for("task_selection"))
    if type == "task_selection":
        TASK = request.form.get('selected_task')
        MODEL = request.form.get('selected_model')
        OVERALL_TIME = time.time()
        SINGLE_TIME = time.time()
        return redirect(url_for("retrieval", annotation_id=1))
    html_file = os.path.basename(type)
    print(type)
    print(html_file)
    return redirect(url_for('index'))


@app.route('/save_and_next_annotation/<int:annotation_id>', methods=['POST'])
def save_and_next_annotation(annotation_id):
    global SQL_FILE, DATABASE, SINGLE_TIME, OVERALL_TIME
    type = request.args.get('type')
    old_time = SINGLE_TIME
    SINGLE_TIME = time.time()
    if type == 'decomposed_sql' or type == 'retrieval':
        global REL_TABLES, REL_EXAMPLES


        logdata = load_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()))
        data_length = len(logdata)
        if len(REL_TABLES) < annotation_id or len(REL_EXAMPLES) < annotation_id:
            REL_TABLES = ['']*data_length
            REL_EXAMPLES = ['']*data_length
        REL_EXAMPLES[annotation_id] = {}
        REL_TABLES[annotation_id] = {}
        if type == 'decomposed_sql':
            annotation = logdata[annotation_id - 1]
            for decomposed in annotation['sql_decomposition']:
                selected_suggested_examples = request.form.getlist('selected_suggested_examples'+decomposed['title'])
                selected_examples = request.form.getlist('selected_examples'+decomposed['title'])
                REL_EXAMPLES[annotation_id][decomposed['title']] = [ast.literal_eval(item) for item in selected_suggested_examples] + [ast.literal_eval(item) for item in selected_examples]
                REL_TABLES[annotation_id][decomposed['title']] = request.form.getlist('selected_suggested_tables'+decomposed['title']) + request.form.getlist('selected_tables'+decomposed['title'])
            url = "decomposed_sql_annotation"
        else:
            selected_suggested_examples = request.form.getlist('selected_suggested_examples')
            selected_examples = request.form.getlist('selected_examples')
            selected_suggested_tables = request.form.getlist('selected_suggested_tables')
            selected_tables = request.form.getlist('selected_tables')


            REL_EXAMPLES[annotation_id] = [ast.literal_eval(item) for item in selected_suggested_examples] + [ast.literal_eval(item) for item in selected_examples]
            if DATABASE == "FIBEN":
                selected_suggested_tables = [x.split('.')[1] if '.' in x else x for x in selected_suggested_tables]
            REL_TABLES[annotation_id] = selected_suggested_tables + selected_tables
            url = "sql_annotation"
        logdata[annotation_id - 1]['time']['retrieval'] = SINGLE_TIME - old_time
        save_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()), logdata)
        return redirect(url_for(url, annotation_id=annotation_id))
    if type == 'decomposed_retrieval':
        sql_in_cte = request.form.getlist('sql_in_cte')
        logdata = load_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()))
        logdata[annotation_id - 1]['sql_in_cte'] = sql_in_cte[0]
        logdata[annotation_id - 1]['time']['decomposition'] = SINGLE_TIME - old_time
        save_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()), logdata)
        return redirect(url_for(type, annotation_id=annotation_id))
    if type == "decomposed_sql_annotation":
        logdata = load_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()))
        annotation = logdata[annotation_id - 1]
        for i in range(len(annotation['sql_decomposition'])):
            selected_option = request.form.get('selected_option'+annotation['sql_decomposition'][i]['title'])
            if selected_option == "adjustment":
                adjustment_text = request.form.get('adjustment_text'+annotation['sql_decomposition'][i]['title'])
                logdata[annotation_id-1]['sql_decomposition'][i]["question"] = adjustment_text
                logdata[annotation_id-1]['sql_decomposition'][i]['adjusted'] = True
            elif selected_option:
                logdata[annotation_id-1]['sql_decomposition'][i]["question"] = selected_option
                logdata[annotation_id-1]['sql_decomposition'][i]['adjusted'] = False
            comment = request.form.get('comment_text'+annotation['sql_decomposition'][i]['title'])
            if comment:
                logdata[annotation_id-1]['sql_decomposition'][i]['comment'] = comment
            else:
                logdata[annotation_id-1]['sql_decomposition'][i]['comment'] = ""
        save_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()), logdata)
        return redirect(url_for("recompose_sql_annotation", annotation_id=annotation_id))
    if type == 'sql_annotation':
        logdata = load_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()))
        selected_option = request.form.get('selected_option')
        if selected_option == "adjustment":
            # Update the adjustment textsolution = "question"
            adjustment_text = request.form.get('adjustment_text')
            logdata[annotation_id-1]["question"] = adjustment_text
            logdata[annotation_id-1]['adjusted'] = True
        elif selected_option:
            logdata[annotation_id-1]["question"] = selected_option
            logdata[annotation_id-1]['adjusted'] = False
        comment = request.form.get('comment_text')
        if comment:
            logdata[annotation_id-1]['comment'] = comment
        else:
            logdata[annotation_id-1]['comment'] = ""
        logdata[annotation_id - 1]['time']['annotation'] = SINGLE_TIME - old_time
        save_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()), logdata)

        if logdata[annotation_id-1]['adjusted']:
            return redirect(url_for("feedback", annotation_id=annotation_id))

        return redirect(url_for("retrieval", annotation_id=annotation_id+1))
    if type == 'feedback':
        global PROMPT_TXT
        selected_option = request.form.get('selected_option')
        if selected_option:
            PROMPT_TXT = selected_option
        logdata = load_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()))
        logdata[annotation_id - 1]['time']['feedback'] = SINGLE_TIME - old_time
        save_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()), logdata)
        return redirect(url_for("retrieval", annotation_id=annotation_id+1))


    return redirect(url_for('index'))


#@app.route('/column_annotation/<int:annotation_id>')
#def column_annotation(annotation_id):
#    data = load_json_data('column')
#    # Fetch the specific annotation based on the ID provided in the URL
#    annotation = data[annotation_id - 1]
#    data_length = len(data)
#    num_annotated=0
#    for i in range(len(data)):
#        if data[i]['annotation'] != "":
#            num_annotated+=1
#    annotation['percentage'] = round(num_annotated / data_length *100,2)
#    db_id = annotation['db_id']
#    table = annotation['table']
#    column = annotation['column']
#    annotation['question'] = f"DB_ID: {db_id}\nTABLE_NAME: {table}\nCOLUMN_NAME:{column}"
#    if "comment" not in annotation.keys():
#        annotation['comment'] = ""
#    return render_template('coming_soon.html', annotation=annotation, annotation_id=annotation_id, data_length=data_length)
#
#@app.route('/verification/<int:annotation_id>')
#def verification(annotation_id):
#    data = load_json_data('bird2')
#    # Fetch the specific annotation based on the ID provided in the URL
#    annotation = data[annotation_id - 1]
#    data_length = len(data)
#    num_annotated=0
#    for i in range(len(data)):
#        if data[i]['annotation'] != []:
#            num_annotated+=1
#    annotation['percentage'] = round(num_annotated / data_length *100,2)
#    annotation['sql'] = sqlparse.format(annotation['sql'], reindent=True, keyword_case='upper')
#    if "comment" not in annotation.keys():
#        annotation['comment'] = ""
#    return render_template('verification.html', annotation=annotation, annotation_id=annotation_id, data_length=data_length)
#
#@app.route('/coming_soon/')
#def coming_soon():
#    return render_template('coming_soon.html')

@app.route('/decomposition/<int:annotation_id>')
def decomposition(annotation_id):
    logdata = load_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()))
    # Fetch the specific annotation based on the ID provided in the URL

    annotation = logdata[annotation_id - 1]
    data_length = len(logdata)
    num_annotated=0
    for i in range(len(logdata)):
        if logdata[i]['question'] != "":
            num_annotated+=1
    annotation['percentage'] = round(num_annotated / data_length *100,2)
    annotation['gold_sql'] = sqlparse.format(annotation['gold-sql'], reindent=True, keyword_case='upper').strip()
    return render_template("decomposition.html", annotation=annotation, annotation_id=annotation_id, data_length=data_length)

@app.route('/decomposed_retrieval/<int:annotation_id>')
def decomposed_retrieval(annotation_id):
    logdata = load_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()))
    # Fetch the specific annotation based on the ID provided in the URL

    annotation = logdata[annotation_id - 1]
    data_length = len(logdata)
    num_annotated=0
    for i in range(len(logdata)):
        if logdata[i]['question'] != "":
            num_annotated+=1
    logdata=pd.DataFrame(logdata)
    annotation['percentage'] = round(num_annotated / data_length *100,2)
    annotation['gold_sql'] = sqlparse.format(annotation['gold-sql'], reindent=True, keyword_case='upper').strip()
    annotations = annotation
    for i in range(len(annotations['sql_decomposition'])):
        most_relevant_examples_ = rank_sentences_more(annotations['sql_decomposition'][i]['sql_embedding'],[logdata['gold-sql'][i] for i in range(data_length) if i != annotation_id-1], [logdata['sql_embedding'][i] for i in range(data_length) if i != annotation_id-1], [logdata['gold-question'][i] for i in range(data_length) if i != annotation_id-1])
        most_relevant_examples = [{'sql':x[0],'question':x[1]} for x in most_relevant_examples_]
        schema = load_json_data(SCHEMA_FILE.format(DATABASE=DATABASE.lower()))
        schema = pd.DataFrame(schema)
        #schema["schema_embedding"] = schema["schema_embedding"].apply(lambda x: np.array(x))
        most_relevant_tables_ = rank_sentences(annotations['sql_decomposition'][i]['sql_embedding'], list(schema['schema']), list(schema["schema_embedding"]))
        most_relevant_tables = Parser(annotations['sql_decomposition'][i]['gold-sql']).tables
        len_tables = len(most_relevant_tables)
        for t in most_relevant_tables_:
            if t[0].split(',')[0] not in most_relevant_tables:
                most_relevant_tables.append(t[0].split(',')[0])
        annotations['sql_decomposition'][i]['gold_sql'] = sqlparse.format(annotations['sql_decomposition'][i]['gold-sql'], reindent=True, keyword_case='upper')
        annotations['sql_decomposition'][i]['suggested_examples']=most_relevant_examples[:5]
        annotations['sql_decomposition'][i]['examples']=most_relevant_examples[5:(min(10, len(most_relevant_examples)-5))]
        annotations['sql_decomposition'][i]['tables']=most_relevant_tables[len_tables:(min(10, len(most_relevant_tables)-len_tables))]
        tables = retrieve_filenames(SCHEMA_FOLDER.format(DATABASE=DATABASE.lower()))
        annotations['sql_decomposition'][i]['suggested_tables']=most_relevant_tables[:len_tables]
        annotations['sql_decomposition'][i]['suggested_tables'] = list(set(annotations['sql_decomposition'][i]['suggested_tables']) & set(tables))
    annotations['percentage'] = round(num_annotated / data_length *100,2)
    annotations['gold_sql'] = sqlparse.format(annotations['gold-sql'], reindent=True, keyword_case='upper')
    return render_template("decomposed_retrieval.html", annotations=annotations, annotation_id=annotation_id, data_length=data_length)

@app.route('/recompose_sql_annotation/<int:annotation_id>')
def recompose_sql_annotation(annotation_id):
    global PROMPT_TXT
    logdata = load_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()))
    # Fetch the specific annotation based on the ID provided in the URL

    annotation = logdata[annotation_id - 1]
    data_length = len(logdata)
    num_annotated=0
    for i in range(len(logdata)):
        if logdata[i]['question'] != "":
            num_annotated+=1
    annotation['percentage'] = round(num_annotated / data_length *100,2)
    annotation['column'] = ['c']
    annotation['gold_sql'] = sqlparse.format(annotation['gold-sql'], reindent=True, keyword_case='upper').strip()
    logdata_=pd.DataFrame(logdata)
    most_relevant_examples_ = rank_sentences_more(annotation['sql_embedding'],[logdata_['gold-sql'][i] for i in range(data_length) if i != annotation_id-1], [logdata_['sql_embedding'][i] for i in range(data_length) if i != annotation_id-1], [logdata_['gold-question'][i] for i in range(data_length) if i != annotation_id-1])
    most_relevant_examples = [{'sql':x[0],'question':x[1]} for x in most_relevant_examples_]
    nl_annotations = {}
    for i in range(len(annotation['sql_decomposition'])):
        nl_annotations[annotation['sql_decomposition'][i]['title']] = annotation['sql_decomposition'][i]['question']
    annotation['options'] = generate_combined_candidate(MODEL, API_KEY, annotation['sql_in_cte'], nl_annotations, most_relevant_examples[1]["question"], PROMPT_TXT)
    return render_template("recompose_sql_annotation.html", annotation=annotation, annotation_id=annotation_id, data_length=data_length)

@app.route('/retrieval/<int:annotation_id>')
def retrieval(annotation_id):
    logdata = load_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()))
    # Fetch the specific annotation based on the ID provided in the URL

    annotation = logdata[annotation_id - 1]
    data_length = len(logdata)
    num_annotated=0
    for i in range(len(logdata)):
        if logdata[i]['question'] != "":
            num_annotated+=1
    logdata=pd.DataFrame(logdata)
    annotation['percentage'] = round(num_annotated / data_length *100,2)
    annotation['gold_sql'] = sqlparse.format(annotation['gold-sql'], reindent=True, keyword_case='upper').strip()
    if 'sql_decomposition' in annotation.keys():
        return redirect(url_for("decomposition", annotation_id=annotation_id))
    most_relevant_examples_ = rank_sentences_more(annotation['sql_embedding'],[logdata['gold-sql'][i] for i in range(data_length) if i != annotation_id-1], [logdata['sql_embedding'][i] for i in range(data_length) if i != annotation_id-1], [logdata['gold-question'][i] for i in range(data_length) if i != annotation_id-1])
    most_relevant_examples = [{'sql':x[0],'question':x[1]} for x in most_relevant_examples_]
    schema = load_json_data(SCHEMA_FILE.format(DATABASE=DATABASE.lower()))
    schema = pd.DataFrame(schema)
    #schema["schema_embedding"] = schema["schema_embedding"].apply(lambda x: np.array(x))

    most_relevant_tables_ = rank_sentences(annotation['sql_embedding'], list(schema['schema']), list(schema["schema_embedding"]))
    most_relevant_tables = Parser(annotation['gold-sql']).tables
    len_tables = len(most_relevant_tables)
    for t in most_relevant_tables_:
        if t[0].split(',')[0] not in most_relevant_tables:
            most_relevant_tables.append(t[0].split(',')[0])

    annotation['suggested_examples']=most_relevant_examples[:5]
    annotation['examples']=most_relevant_examples[5:(min(10, len(most_relevant_examples)-5))]
    annotation['tables']=most_relevant_tables[len_tables:(min(10, len(most_relevant_tables)-len_tables))]
    annotation['suggested_tables']=most_relevant_tables[:len_tables]
    return render_template("retrieval.html", annotation=annotation, annotation_id=annotation_id, data_length=data_length)
#@app.route('/correction/')
#def correction():
#    annotation_id = 1
#    annotation = {'options': ['BIRD', 'SPIDER', 'BEAVER'], 'percentage' :0.5, 'data':"What are the most relevant buildings at MIT?"}
#    annotation['suggested_examples']=['example1', 'example2', 'example3']
#    annotation['examples']=['example11', 'example12', 'example3']
#    annotation['tables']=['table11', 'table12', 'table13']
#    annotation['suggested_tables']=['table1', 'table2', 'table3']
#    return render_template('correction.html', annotation=annotation, annotation_id=annotation_id, data_length=10)
@app.route('/review/')
def review():
    logdata = pd.read_json(SQL_FILE.format(DATABASE=DATABASE.lower())+".json")
    """
    Computes BLEU, ROUGE, METEOR, and BERTScore for two lists of NL queries.
    Returns a list of scores for each pair.
    """
    logdata = logdata[logdata['question']!=""]
    generated_list = logdata["question"]
    reference_list = logdata["gold-question"]
    if len(reference_list) != len(generated_list):
        raise ValueError("Both lists must have the same length.")

    results = {"BLEU": [], "ROUGE": [],  "BERTScore": []}

    for ref, gen in zip(reference_list, generated_list):
        if not ref or not gen:
            scores = {"BLEU": 0.0, "ROUGE": 0.0,  "BERTScore": 0.0}
        else:

            scores = evaluate_nl_accuracy(ref, gen)

        results['BLEU'].append(float(scores['BLEU']))
        results['ROUGE'].append(float(scores['ROUGE']))
        results['BERTScore'].append(float(scores['BERTScore']))

    result = {"adjusted": sum(logdata[logdata["adjusted"]==True]["adjusted"]), "annotated": len(generated_list),"bleu": round(np.mean(results['BLEU'])*100,2), "rouge": round(np.mean(results['ROUGE'])*100,2),  "bert": round(np.mean(results['BERTScore'])*100, 2)}
    print(result)
    return render_template('review.html', annotation=result)
@app.route('/feedback/<int:annotation_id>')
def feedback(annotation_id):
    global PROMPT_TXT,API_KEY, MODEL, SQL_FILE, DATABASE
    logdata = load_json_data(SQL_FILE.format(DATABASE=DATABASE.lower()))
    data_length = len(logdata)
    annotation = logdata[annotation_id-1]
    annotation['prompt'] = PROMPT_TXT
    annotation['new_prompt'] = generate_improved_prompt(MODEL, API_KEY, PROMPT_TXT, logdata[annotation_id-1]['options'], logdata[annotation_id-1]['question'],logdata[annotation_id-1]['comment'], DATABASE)
    annotation["options"] = "\n".join(f"• {opt}" for opt in logdata[annotation_id-1]['options'])
    return render_template('feedback.html', annotation=annotation, annotation_id=annotation_id, data_length=data_length)
@app.route("/upload")
def upload():
    annotation={}
    annotation['options'] = DATA['datasets']
    annotation['temp'] = DATABASE
    return render_template("upload.html", annotation=annotation)

if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=True)
