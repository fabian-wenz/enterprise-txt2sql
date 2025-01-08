from openai import OpenAI
import os
import pandas as pd
import re
from sql_metadata import Parser

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


# Initialize OpenAI API with your key
client = OpenAI(api_key="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")

# Read the SQL queries from JSON file into a dataframe
df = pd.read_json("bird.json")#("sql48clean.json")

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
                "content": f"""Your task is to generate 4 different natural language options for the given SQL query. 
                The goal is for the user to later select the most appropriate natural language question from these options. 
                Each option should:
                1. Clearly describe the purpose of the SQL query.
                2. Focus on the columns, filtering conditions, and overall intent of the query.
                3. Be distinct from the others while staying accurate to the query, especially in the overall order and amount of columns.
                
                Below is an example of a SQL query and its gold standard natural language description, followed by the task:
                
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
df.to_json("bird2.json", orient="records")
