import json
import os
from time import sleep

import pandas as pd
from openai import OpenAI
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from pyrate_limiter import Duration, RequestRate, Limiter
from tqdm import tqdm
import backoff

N = 3
SUBSET = 200

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

limit = RequestRate(100, Duration.MINUTE)
limiter = Limiter(limit)

MAPPINGS = "goby_schema_mappings.csv"

CLASSES = [
    "STARTDATE",
    "WEBSITE",
    "LOC1.STATE",
    "PHONE",
    "TITLE",
    "DATETIME",
    "CALL_TO_ACTION_URL",
    "DESCRIPTION",
    "LOC1.ADDRESS",
    "ENDDATE",
    "VENUE_NAME",
    "LOC1.CITY",
    "LOC1.LON",
    "HOURS",
    "EMAIL",
    "LOC1.FULL_ADDRESS",
    "LOC1.ZIP",
    "IMAGE1",
    "TIMERANGE",
    "LOC1.LAT",
    "PRICE_RANGE",
    "STARTTIME",
    "DATERANGE",
    "ENDTIME",
    "REVIEWS",
    "ARTIST",
    "ENDDATETIME",
    "DURATION",
    "RATING",
    "DIRECTIONS",
    "LOC1.COUNTY",
    "STARTDATETIME",
    "TOTAL_NUMBER_OF_RATINGS",
    "LOC1.COUNTRY",
]

answers = pd.read_csv(MAPPINGS, header=[0], index_col=[0, 1]).squeeze()

# filter unlabeled columns ('NO_DISPLAY') and those with very low support
answers = answers[answers.isin(CLASSES)]


def custom_formatter(value):
    if isinstance(value, str):
        return f"{value[:25] if len(value) > 25 else value}"
    elif isinstance(value, float):
        return f"{value:.3g}"
    else:
        return value


def prompt(df, col_name):
    serialized_table = (
        df.map(custom_formatter).head(n=N).to_string(max_colwidth=30, max_cols=25)
    )
    return f"""Consider the column `{col_name}` in the data sample below.
    {serialized_table}

    Select the class a column called `{col_name}` with values `{list(df[col_name].unique()[:3])}` from the list:
    
    {CLASSES}

    Reply with only one word from the list."""

@backoff.on_exception(backoff.expo,
                      ValueError,
                      max_time=15)
def response(prompt, n_tokens, temperature=0.1, model="gpt-3.5-turbo"):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Be a helpful, accurate assistant for data discovery and exploration.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=n_tokens,
        temperature=temperature,
    )

    return completion.choices[0].message.content


def predict_column(wrapper_id, col_name):
    filename = f"wrappers/goby_{wrapper_id}.csv"
    wrapper = pd.read_csv(filename, header=[0])
    message = prompt(wrapper, col_name)
    return response(message, 10)


def run_experiment():
    subset = answers[:SUBSET]
    subset = shuffle(subset, random_state=0)
    predictions = []
    for index, label in tqdm(subset.items(), total=subset.size):
        wrapper_id, field_name = index
        answer = predict_column(wrapper_id, field_name)
        predictions.append(answer)
        sleep(0.1)

    results = pd.DataFrame({"labels": subset.values, "preds": predictions})
    results.to_parquet("artifacts/goby_col_preds.parquet")

    report = classification_report(results.labels, results.preds)
    print(report)


if __name__ == "__main__":
    run_experiment()
