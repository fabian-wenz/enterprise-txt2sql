from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def get_cosine_similarity(nl_query, sql_query):

    nl_embedding = get_embedding(nl_query)
    sql_embedding = get_embedding(sql_query)

    # Cosine Similarity
    cosine_similarity = torch.nn.functional.cosine_similarity(nl_embedding, sql_embedding)
    return cosine_similarity


def rank_sentences2(reference_sentence, sentence_list):
# Load the model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Encode reference sentence and detach
    ref_embedding = model.encode(reference_sentence, convert_to_tensor=True).detach().cpu().numpy()

    # Encode list of sentences and detach
    sentence_embeddings = model.encode(sentence_list, convert_to_tensor=True).detach().cpu().numpy()

    # Compute cosine similarities
    similarities = np.dot(sentence_embeddings, ref_embedding) / (
        np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(ref_embedding)
    )

    # Sort sentences by similarity score
    ranked_sentences = sorted(zip(sentence_list, similarities), key=lambda x: x[1], reverse=True)

    return ranked_sentences


def rank_sentences(ref_embedding, sentence_list, sentence_embeddings):

    # Compute cosine similarities
    similarities = np.dot(sentence_embeddings, ref_embedding) / (
        np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(ref_embedding)
    )

    # Sort sentences by similarity score
    ranked_sentences = sorted(zip(sentence_list, similarities), key=lambda x: x[1], reverse=True)

    return ranked_sentences

def rank_sentences_more(ref_embedding, sentence_list, sentence_embeddings, nl_questions):
    """
    Ranks sentences (SQL queries) by similarity and retrieves the corresponding NL questions.

    :param ref_embedding: The embedding of the reference SQL query.
    :param sentence_list: List of SQL queries.
    :param sentence_embeddings: Corresponding embeddings for the SQL queries.
    :param nl_questions: List of corresponding NL questions.
    :return: List of tuples (SQL query, NL question, similarity score), sorted by similarity.
    """
    if sentence_embeddings == []:
        return []
    # Compute cosine similarities
    similarities = np.dot(sentence_embeddings, ref_embedding) / (
        np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(ref_embedding)
    )

    # Sort SQL queries by similarity, keeping track of their NL questions
    ranked_sentences = sorted(zip(sentence_list, nl_questions, similarities), key=lambda x: x[2], reverse=True)

    return ranked_sentences

if __name__ == '__main__':
    # Example: Compare SQL and NL
    #nl_query = "Get all customers who made a purchase last month"
    #sql_query = "SELECT * FROM customers WHERE purchase_date >= DATE_SUB(NOW(), INTERVAL 1 MONTH)"
    #
    #nl_embedding = get_embedding(nl_query)
    #sql_embedding = get_embedding(sql_query)
    #
    ## Cosine Similarity
    #cosine_similarity = torch.nn.functional.cosine_similarity(nl_embedding, sql_embedding)
    #print(f"Similarity Score: {cosine_similarity.item()}")
    # Example Usage
    reference = "Find all customers who made a purchase last month"
    sentences = [
        "Retrieve customers who bought something last month",
        "Show me the employees who joined last year",
        "List all customers with recent transactions",
        "Get all customers from the database",
        "Find all orders from the past month"
    ]

    ranked = rank_sentences(reference, sentences)

    # Print the results
    for sentence, score in ranked:
        print(f"Score: {score:.4f} | {sentence}")
