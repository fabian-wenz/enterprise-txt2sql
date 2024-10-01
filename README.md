# LLMs for Enterprise Data?
Exploring the potential of LLMs for exploring structured data in the enterprise.

The structure of data input:
./data/{data_base_name/
with the following sub-folders and files
./data/{data_base_name}/schema/
containing all tables as the csv-filenames and three columns (COLUMN_NAME,DATA_TYPE,KEY_TYPE)
./data/{data_base_name}/embdes/
as a folder for the created embedding for all questions with the top k tables
./data/{data_base_name}/[data|database]
as access to verify the tables with the underlying actual data
./data/{data_base_name}/output/
as folder to save the predicted queries as csv files
./data/{data_base_name}/examples.json
json file containing (sql,question)-pairs to provide them to the prompt  as additional information
./data/{data_base_name}/queries.json
json file containing (sql,question)-pairs to ask the questions th LLM-model
./data/{data_base_name}/tables.csv
csv files containing a schema-column containing (tablenames,column1,...) used for the embedding

1. call retrieve.py to get embeddings
2. call get_gpt_preds
3. call get_evaluation
