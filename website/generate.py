from openai import OpenAI
import os
import pandas as pd
import re
from sql_metadata import Parser
import pandas as pd

PROMPT = """Your task is to generate 4 different natural language options for the given SQL query. 
        The goal is for the user to later select the most appropriate natural language question from these options. 
        Each option should:
        1. Clearly describe the purpose of the SQL query.
        2. Focus on the columns, filtering conditions, and overall intent of the query.
        3. Be distinct from the others while staying accurate to the query, especially in the overall order and amount of columns.
        
        Below is an example of a SQL query and its gold standard natural language description, followed by the task:"""
def clean(option):
    words = option.split()
    if not words:
        return option
    first = words[0]
    if len(first) < 5 and (re.search(r'[\d\.\-]', first) or not first.isalpha()):
        return " ".join(words[1:])
    return option

def retrieve_most_relevant_example(sql, examples, index):
    tables = set(Parser(sql.upper()).tables)
    tables = {item.split('.', 1)[-1] for item in tables}
    check_queries = {}
    for i in range(len(examples)):
        table_examples = {item.split('.', 1)[-1] for item in Parser(examples[i].upper()).tables}
        num = len(tables.intersection(table_examples))
        if len(table_examples) != len(tables):
            num -= abs(len(tables)-len(table_examples))/len(tables)
        check_queries[i] = num
    sorted_queries = sorted(check_queries.items(), key=lambda x: x[1], reverse=True)
    sorted_queries = [x for x in sorted_queries if x[0] !=index]
    return sorted_queries[0:min(5, len(examples))]

def create_examples(df, examples):
    prompt_form = "### Example {example_num}:\n"\
                "SQL Query:\n"\
                "{sql}\n"\
                "Gold Standard NL Question:\n"\
                "{question}\n\n"
    prompt = ""
    for i in range(len(examples)):
        prompt += prompt_form.format(example_num=i, sql=df['sql'][examples[i][0]], question=df['question'][examples[i][0]])
    return prompt

def create_examples(examples):
    prompt_form = "### Example {example_num}:\n"\
                "SQL Query:\n"\
                "{sql}\n"\
                "Gold Standard NL Question:\n"\
                "{question}\n\n"
    prompt = ""
    for i in range(len(examples)):
        prompt += prompt_form.format(example_num=i, sql=examples[i]['sql'], question=examples[i]['question'])
    return prompt

def create_tablestatements(tables, tables_folder, db_id):
    prompt_form = "###  Relevant Table {example_num}:\n"\
                "CREATE TABLE {table_name}(\n"\
                "{columns}"\
                ");\n\n"
    prompt = ""
    for i in range(len(tables)):
        table = tables[i].split(',')
        if db_id:
            table_file_name= db_id.upper() + '-' + table[0].upper()
        else:
            table_file_name = table[0].upper()
        table_attr = pd.read_csv(tables_folder + table_file_name + '.csv')
        columns = ""
        for i in range(len(table_attr)):
            columns = "   " + table_attr['COLUMN_NAME'][i]
            if table_attr['DATA_TYPE'].notnull()[i]:
                columns += " " + table_attr['DATA_TYPE'][i]
            if table_attr['PKEY'].notnull()[i]:
                columns += " " + table_attr['PKEY'][i]
            if table_attr['FKEY'].notnull()[i]:
                columns += " " + table_attr['FKEY'][i]
            columns += ",\n"
        columns = columns[:-2]
        prompt += prompt_form.format(example_num=i, table_name=table[0], columns=columns)
    return prompt


def generate_candidates(api_key, sql_file):
    # Initialize OpenAI API with your key
    client = OpenAI(api_key=api_key)

    # Read the SQL queries from JSON file into a dataframe
    df = pd.read_json(sql_file)#("sql48clean.json")

    df['options'] = [[] for _ in range(1, len(df) + 1)]
    df['annotation'] = [[] for _ in range(1, len(df) + 1)]
    # Iterate through each row in the dataframe
    for i in range(50):
        create_examples(df, retrieve_most_relevant_example(df['sql'][i], df['sql'], i))


        # Make the API call to generate natural language questions
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,  # Zero temperature ensures correctness and consistency
            messages=[
                {
                    "role": "system",
                    "content": f"""{PROMPT}
                    
                    {create_examples}
                    
                    SQL Query:"""
                },
                {
                    "role": "user",
                    "content": df['sql'][i]  # SQL query from the dataframe
                }
            ]
        )

        # Extract the generated questions from the response
        res = response.choices[0].message.content

        # Split the result into individual questions based on numbering format (e.g., '1. ', '2.
        if '.' in res:
            questions = re.split(r'\d+\.\s', res)
        if ':' in res:
            questions = re.split(r'\d+:\s', res)
        # Remove any empty strings from the split
        options = []
        for question in questions:
            if len(question) > 10:
                options.append(question)

        # Store the questions back into the dataframe under the 'options' column
        df.at[i, 'options'] = options
        print(options)

    # Save the modified dataframe with questions back to the JSON file
    df = df[['sql', 'question', 'options', 'annotation', 'db_id']]
    df.to_json(sql_file, orient="records")

def create_formated_output(rows, headers):
    if len(rows) > 0 and len(headers)>0 and headers[0] != "ERROR":
        rows = rows[:min(len(rows), 5)]
        # Combine headers and rows to compute max width for each column
        all_rows = [headers] + [list(map(str, row)) for row in rows]
        col_widths = [max(len(str(cell)) for cell in col) for col in zip(*all_rows)]

        # Format rows
        def format_row(row):
            return " | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths))

        return "\n".join([format_row(headers)] + ["-" * sum(col_widths) + "-" * (3 * (len(headers) - 1))] + [format_row(row) for row in rows])
    return ""

def generate_candidate(model, api_key, prompt,prompt_txt, sql, tables, examples, db_id, db_set, rows, headers):
    # Initialize OpenAI API with your key
    client = OpenAI(api_key=api_key)
    # Iterate through each row in the dataframe

    example_prompt = create_examples(examples)
    output_prompt = create_formated_output(rows, headers)
    table_statements = create_tablestatements(tables, './data/' + db_set.lower() + '/schema/', db_id)


    # Make the API call to generate natural language questions
    response = client.chat.completions.create(
        model=model,
        temperature=0,  # Zero temperature ensures correctness and consistency
        messages=[
            {
                "role": "system",
                "content": prompt.format(examples=example_prompt, schema=table_statements, prompt_text=prompt_txt) + output_prompt
            },
            {
                "role": "user",
                "content": sql  # SQL query from the dataframe
            }
        ]
    )

    # Extract the generated questions from the response
    res = response.choices[0].message.content

    options = [line.strip() for line in res.splitlines() if line.strip() and not "OPTION" in line.upper() and not ":" in line]
    options = [clean(option) for option in options]
    # Store the questions back into the dataframe under the 'options' column

    print(options)
    return options

def insert_nl_annotations(sql, annotations):
    for title in annotations:
        if f"WITH {title} AS (\n" in sql:
            sql = sql.replace(f"WITH {title} AS (\n", "-- " + annotations[title] + "\n"+ f"{title} AS (\n")
        elif f"{title} AS (\n" in sql:
            sql = sql.replace(f"{title} AS (\n", "-- " + annotations[title] + "\n"f"{title} AS (\n")
        elif title == "main":
            comment = annotations[title]
            match = re.search(r"\)\s*SELECT", sql, re.IGNORECASE | re.DOTALL)
            if match:
                insert_index = match.start() + 1  # right after the last )
                sql = sql[:insert_index] + f"\n-- {comment}\n" + sql[insert_index:]
            else:
                # No CTE found, just prepend the comment
                sql = "-- {comment}\n{sql}"
    return sql


def generate_combined_candidate(model, api_key, sql_in_cte, nl_annotations, example, priorities, rows, headers):
    """
    Function to generate a natural language explanation for the given SQL queries.
    This function combines both SQL queries and uses an LLM to generate the explanation.
    """

    sql_in_cte = insert_nl_annotations(sql_in_cte, nl_annotations)
    expected_output = "### Output from above SQL:\n" + create_formated_output(rows, headers) if rows else ""
    # Construct the prompt with task definition and SQL queries
    prompt = f"""
    ### Task:
    You are given a SQL query with a common table expression (CTE), where parts of the query are already annotated in natural language (NL). Your task is to generate **four different natural language questions** that describe the entire SQL logic, focusing on the output.
    
    {priorities}
    
    ### SQL Query with CTE and NL annotations:
    {sql_in_cte}
    
    {expected_output}
    
    ### Request:
    Provide **four different question-based NL options** that describe the data retrieved by the SQL query. These options should focus on the **output** and avoid mentioning SQL structures like CTEs, joins, or unions. Formulate each option as a clear and concise question, similar to the following example:
    
    Example option format:
    {example}
    
    Generate your options in a similar question style."""
    cte_summary = "\n".join([f"--{title}: {nl_annotations[title]}" for title in nl_annotations])
    prompt = f"""
    ### Task:
    You are given a SQL query composed of multiple Common Table Expressions (CTEs), each with an associated natural language (NL) annotation. Your goal is to **compose four different, clear natural language question** that captures the meaning of the **ffinal result of the SQL query.**, integrating all previous annotations, the intent and constraints from all CTEs.

    {priorities}
    
    ---
    
    ### Please summarize these intermediate steps into one option:
    {cte_summary}  
      
    
    ---
    
    ### SQL Query (with optional annotations):
    {sql_in_cte}
    
    ---
    
    ### Your Task:
    Write four diverse, natural language questions that reflect the final output of this query and joins the nl annotations. Avoid SQL terms like "CTE", "join", or "alias". Each question should describe what the query returns — based on the annotations and SQL logic.
    
    ### Output format:
    1. [question one]
    2. [question two]
    3. [question three]
    4. [question four]
    """
    print(prompt)
    # Initialize OpenAI API with your key
    client = OpenAI(api_key=api_key)
    # Iterate through each row in the dataframe

    # Make the API call to generate natural language questions
    response = client.chat.completions.create(
        model=model,
        temperature=0,  # Zero temperature ensures correctness and consistency
        messages=[
            {
                "role": "user",
                "content": prompt + create_formated_output(rows, headers) if rows else ""# SQL query from the dataframe
            }
        ]
    )

    # Extract the generated questions from the response
    res = response.choices[0].message.content

    options = [line.strip() for line in res.splitlines() if line.strip() and not "OPTION" in line.upper() and not ":" in line]

    # Store the questions back into the dataframe under the 'options' column
    options = [clean(option) for option in options]
    options = [option for option in options if len(option)>3]
    print(options)
    return options



def generate_improved_prompt(model, api_key, prompt_text, options, selected_option, user_statement, user_comment):
    """
    Calls GPT to generate a better version of the given prompt while ensuring placeholders {examples} and {schema}
    remain intact.
    """
    # Initialize OpenAI API
    client = OpenAI(api_key=api_key)
    prompt_text = prompt_text
    # Construct the GPT request
    response = client.chat.completions.create(
        model=model,
        temperature=0,  # Slight creativity while maintaining accuracy
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in prompt engineering. Your task is to refine and add a bullet point to a given prompt "
                    "while ensuring clarity, effectiveness, and usability. "
                    "Maintain the structure of the original prompt, but you must analyze and improve the enumerated list. "
                    "Modify existing points if needed and add at least one additional refinement to make it more specific and structured. "
                    "Ensure the prompt is highly effective for guiding the user. "
                    "Do not change the intent or placeholders in the original prompt."
                    "Refine the enumerated list of the upcoming given prompt while keeping the structure intact. Ensure the improved version is clearer, "
                    "more structured, and useful for guiding the user in selecting the best natural language question.\n\n"
                    "**Available Options from the upcoming prompt:**\n{options}\n\n"
                    "**User's Written and Selected Statement:**\n{user_statement}\n\n"
                    "**User's Comment (Why Other Options Were Not Chosen):**\n{user_comment}\n\n"
                ).format(
                    options="\n".join(f"- {opt}" for opt in options),
                    user_statement=user_statement,
                    user_comment=user_comment
                )
            },
            {
                "role": "user",
                "content": (
                    "**Original Prompt:**\n{original_prompt}\n\n"
                ).format(
                    original_prompt=prompt_text
                )
            }
        ]
    )

    # Extract the improved prompt from GPT response
    improved_prompt = response.choices[0].message.content
    start_index = improved_prompt.find("Each option should:")

    if start_index != -1:
        improved_prompt =improved_prompt[start_index:]
    return improved_prompt

