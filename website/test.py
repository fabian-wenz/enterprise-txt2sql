import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score
import pandas as pd
import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
# === Simple Tokenizers ===
def simple_sent_tokenize(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

# === Evaluation Function ===
def evaluate_nl_accuracy(reference, generated):
    bleu = sentence_bleu([reference.split()], generated.split(), smoothing_function=SmoothingFunction().method1)
    rouge = Rouge().get_scores(generated, reference)[0]["rouge-1"]["f"]
    P, R, F1 = score([generated], [reference], lang="en", rescale_with_baseline=True, verbose=False)
    return {
        "BLEU": bleu,
        "ROUGE-1": rouge,
        "BERTScore": F1.item()
    }

# === Inputs ===
documents = {
    "Studies show cats sleep 70% of their lives.": 1,
    "The internet was first conceptualized in the 1960s...": 2,
    "Cats are curious and observant animals.": 3,
    "Quantum mechanics is a fundamental theory in physics...": 4,
    "Cats were first domesticated in the Near East around 7500 BC.": 5
}
documents = {
    "After mashing , the beer wort is boiled with hops ( and other flavourings if used ) in a large tank known as a \" copper \" or brew kettle – though historically the mash vessel was used and is still in some small breweries . ":1,
    "The boiling process is where chemical reactions take place , including sterilization of the wort to remove unwanted bacteria , releasing of hop flavours , bitterness and aroma compounds through isomerization , stopping of enzymatic processes , precipitation of proteins , and concentration of the wort . ":2,
    "Finally , the vapours produced during the boil volatilise off - flavours , including dimethyl sulfide precursors . The boil is conducted so that it is even and intense – a continuous \" rolling boil \" . ":3,
    "The boil on average lasts between 45 and 90 minutes , depending on its intensity , the hop addition schedule , and volume of water the brewer expects to evaporate . ":4,
    "At the end of the boil , solid particles in the hopped wort are separated out , usually in a vessel called a \" whirlpool \".":5,
}
model_response = """
Some of the most surprising facts about cats include that they spend roughly 70% of their lives sleeping, as indicated in various studies. This equates to approximately 13 to 14 hours a day. Moreover, despite being domestic animals in many households worldwide, cats retain curious and observant personalities, allowing them to learn and adapt to their surroundings efficiently. Additionally, cats have been companions to humans for a very long time; evidence indicates that they were first domesticated around 7600 BC in the Near East.
"""
model_response = """
Hops are added to the brewing process during the boiling stage, where they are boiled with the wort to extract bitterness, flavor, and aroma. Early additions during the boil increase bitterness, while hops added later or near the end enhance aroma and flavor. This timing allows hops to undergo isomerization, releasing essential compounds that balance the sweetness of the malt and contribute to the beer's overall character.
"""

# === Tokenize ===
reference_sents = list(documents.keys())
candidate_sents = simple_sent_tokenize(model_response)

# === Evaluate ===
results = []
for cand in candidate_sents:
    # Find the best matching reference for each generated sentence
    best_score = None
    best_ref = None
    for ref in reference_sents:
        score_dict = evaluate_nl_accuracy(ref, cand)
        if best_score is None or score_dict["BERTScore"] > best_score["BERTScore"]:
            best_score = score_dict
            best_ref = ref
    results.append({
        "Generated": cand,
        "Best Reference": best_ref,
        **best_score
    })

# === Display as DataFrame ===
df = pd.DataFrame(results)
print(df.to_string(index=False))
reference_full = " ".join(reference_sents)
print(candidate_sents)
generated_full = " ".join([candidate_sents[1]]+candidate_sents[3:])
#generated_full = " ".join(candidate_sents)

overall_score = evaluate_nl_accuracy(reference_full, generated_full)

# Print or append it
print("\n=== Overall Score ===")
print(f"BLEU:       {overall_score['BLEU']:.4f}")
print(f"ROUGE-1:    {overall_score['ROUGE-1']:.4f}")
print(f"BERTScore:  {overall_score['BERTScore']:.4f}")
