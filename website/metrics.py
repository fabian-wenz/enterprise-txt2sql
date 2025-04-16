from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import score

def evaluate_nl_accuracy(reference, generated):
    bleu = sentence_bleu([reference.split()], generated.split())
    rouge = Rouge().get_scores(generated, reference)[0]["rouge-l"]["f"]
    P, R, F1 = score([generated], [reference], lang="en", rescale_with_baseline=True)

    return {
        "BLEU": bleu,
        "ROUGE": rouge,
        "BERTScore": F1.item()
    }

# Example Usage
reference = "How many employees are older than 30?"
generated = "Find the total number of employees who are over 30."

scores = evaluate_nl_accuracy(reference, generated)
print(scores)
