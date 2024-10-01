"""
schema_as_table
---------
Reads the single-column text schema files saved as CSV files under 
inpath=data/views/schema/and converts them into proper tables and 
saves the converted schemas under outpath=data/views/schema_as_table/ 
again in CSV format.

Example
--------

"""
import sys
import os
import pandas as pd


def schema_as_table(inpath='../../data/views/schema/', outpath='../../data/views/schema_as_table/'):
    files = os.listdir(inpath)
    for file in files:
        if file.endswith(".csv"):
            schema_file = inpath + file
            df_schema = pd.read_csv(schema_file, header=None)
            df_schema = df_schema.drop(1)  # delete the "header" line
            column_names = [h.strip() for h in df_schema[0][0].split(';')]
            df_schema_new = pd.DataFrame(list(df_schema[0][1:].apply(lambda row: [c.strip() for c in row.split(';')])),
                                         columns=column_names)
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            new_schema_file = outpath + file
            df_schema_new.to_csv(new_schema_file, index=False)


if __name__ == '__main__':
    sys.exit(schema_as_table())

