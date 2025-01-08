import textwrap
from flask import Flask, render_template, request, redirect, url_for, json
import sqlparse


app = Flask(__name__)

def load_json_data(file_name):
    with open(file_name + '.json', 'r') as f:
        data = json.load(f)
    return data

def save_json_data(file_name, data):
    with open(file_name + '.json', 'w') as f:
        json.dump(data, f, indent=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/submit", methods=["POST"])
def submit():
    # Retrieve data from the form
    api_key = request.form.get("api_key")
    username = request.form.get("username")
    sql_file = request.files.get("sql_file")

    # Save the SQL file locally (optional)
    if sql_file:
        sql_file.save(f"./uploaded_files/{sql_file.filename}")

    # Process the API key, username, and SQL file as needed
    print(f"API Key: {api_key}")
    print(f"Username: {username}")
    print(f"SQL File: {sql_file.filename if sql_file else 'No file uploaded'}")

    # Redirect back to the homepage after submission
    return redirect(url_for("index"))

@app.route('/sql_annotation/<int:annotation_id>')
def sql_annotation(annotation_id):
    data = load_json_data('sql')
    # Fetch the specific annotation based on the ID provided in the URL
    annotation = data[annotation_id - 1]
    data_length = len(data)
    num_annotated=0
    for i in range(len(data)):
        if data[i]['question'] != "":
            num_annotated+=1
    annotation['percentage'] = round(num_annotated / data_length *100,2)
    annotation['sql'] = sqlparse.format(annotation['sql'], reindent=True, keyword_case='upper')
    if "comment" not in annotation.keys():
        annotation['comment'] = ""
    return render_template('sql_annotation.html', annotation=annotation, annotation_id=annotation_id, data_length=data_length)

@app.route('/save_and_next_annotation/<int:annotation_id>', methods=['POST'])
def save_and_next_annotation(annotation_id):
    type = request.args.get('type')
    if type == 'verification':
        data = load_json_data('bird2')
    else:
        data = load_json_data(type)
    annotation = data[annotation_id - 1]

    # Update the selected option
    selected_option = request.form.get('selected_option')
    solution = "annotation"
    url = type + "_" + solution
    if type == 'sql':
        solution = "question"
    elif type == 'verification':
        solution = "annotation"
        url = "verification"
    if selected_option == "adjustment":
        # Update the adjustment text
        adjustment_text = request.form.get('adjustment_text')
        annotation[solution] = adjustment_text
        annotation['adjusted'] = True
    elif selected_option:
        annotation[solution] = selected_option
        annotation['adjusted'] = False
    comment = request.form.get('comment_text')
    if comment:
        annotation['comment'] = comment
    else:
        annotation['comment'] = ""
    '''if type == "api_mapping":
        if solution not in annotation.keys():
            annotation[solution] = {}
        for key in annotation['extracted_tables']:
            selected_option = request.form.get(f'selected_option_{key}')
            print(f'selected_option_{key}')
            print(selected_option)
            if key not in annotation[solution].keys():
                annotation[solution][key] = {}
            if selected_option:
                if "adjustment" in selected_option:
                    adjustment_text = request.form.get(f'adjustment_text_{key}')
                    annotation[solution][key]['api'] = adjustment_text
                    annotation[solution][key]['adjusted'] = True
                elif selected_option:
                    annotation[solution][key]['api'] = selected_option
                    annotation[solution][key]['adjusted'] = False
            comment = request.form.get(f'comment_text_{key}')
            if comment:
                annotation[solution][key]['comment'] = comment
            else:
                annotation[solution][key]['comment'] = ""'''

    # Save the updated data back to the JSON file
    print('_______________________________________')
    print(type)
    print('_______________________________________')
    if type == 'verification':
        save_json_data('bird2', data)
    else:
        save_json_data(type, data)

    # Redirect to the next annotation, or stay on the same page if it's the last one
    next_annotation_id = annotation_id + 1
    if next_annotation_id > len(data):
        next_annotation_id = annotation_id  # Stay on the current page if there are no more annotations

    return redirect(url_for(url, annotation_id=next_annotation_id))


@app.route('/column_annotation/<int:annotation_id>')
def column_annotation(annotation_id):
    data = load_json_data('column')
    # Fetch the specific annotation based on the ID provided in the URL
    annotation = data[annotation_id - 1]
    data_length = len(data)
    num_annotated=0
    for i in range(len(data)):
        if data[i]['annotation'] != "":
            num_annotated+=1
    annotation['percentage'] = round(num_annotated / data_length *100,2)
    db_id = annotation['db_id']
    table = annotation['table']
    column = annotation['column']
    annotation['question'] = f"DB_ID: {db_id}\nTABLE_NAME: {table}\nCOLUMN_NAME:{column}"
    if "comment" not in annotation.keys():
        annotation['comment'] = ""
    return render_template('column_annotation.html', annotation=annotation, annotation_id=annotation_id, data_length=data_length)

@app.route('/verification/<int:annotation_id>')
def verification(annotation_id):
    data = load_json_data('bird2')
    # Fetch the specific annotation based on the ID provided in the URL
    annotation = data[annotation_id - 1]
    data_length = len(data)
    num_annotated=0
    for i in range(len(data)):
        if data[i]['annotation'] != []:
            num_annotated+=1
    annotation['percentage'] = round(num_annotated / data_length *100,2)
    annotation['sql'] = sqlparse.format(annotation['sql'], reindent=True, keyword_case='upper')
    if "comment" not in annotation.keys():
        annotation['comment'] = ""
    return render_template('verification.html', annotation=annotation, annotation_id=annotation_id, data_length=data_length)

@app.route('/api_mapping_annotation/<int:annotation_id>')
def api_mapping_annotation(annotation_id):
    data = load_json_data('api_mapping')
    # Fetch the specific annotation based on the ID provided in the URL
    annotation = data[annotation_id - 1]
    data_length = len(data)
    num_annotated=0
    total = 0
    for i in range(len(data)):
        total += len(data[i]['extracted_tables'])
        for key in data[i]['extracted_tables'].keys():
            if 'annotation' in data[i].keys() and key in data[i]['annotation'].keys() and 'api' in data[i]['annotation'][key]:
                if data[i]['annotation'][key]['api'] != "":
                    num_annotated+=1
    annotation['percentage'] = round(num_annotated / total *100,2)
    if "annotation" not in annotation.keys():
        annotation['annotation'] = {}

    for key in annotation['extracted_tables'].keys():
        annotation['extracted_tables'][key] = '[' + textwrap.fill(", ".join(annotation['extracted_tables'][key]), width=80) + ']'
        if key not in annotation['annotation'].keys():
            annotation['annotation'][key] = {'comment':"", 'api':"", 'adjuested':False}
        else:
            if "comment" not in annotation['annotation'][key].keys():
                annotation['annotation'][key]['comment'] = ""

            if "api" not in annotation['annotation'][key].keys():
                annotation['annotation'][key]['api'] = ""

            if "adjusted" not in annotation['annotation'][key].keys():
                annotation['annotation'][key]['adjusted'] = False
    return render_template('api_mapping_annotation.html', annotation=annotation, annotation_id=annotation_id, data_length=data_length)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
