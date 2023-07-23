
# import torch
# import pandas as pd
# from sentence_transformers import SentenceTransformer, util
# from transformers import BertTokenizer

# # Load the SentenceTransformer model for sentence embeddings
# model = SentenceTransformer('bert-base-nli-mean-tokens')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# max_length = 128

# # Read the CSV file with 'text1' and 'text2' columns
# data = pd.read_csv('Precily_Text_Similarity.csv')
# data=data.iloc[:3,:].copy()
# def calculate_similarity(text1, text2):
#     # Encode the input texts into sentence embeddings
#     embeddings1 = model.encode(text1, convert_to_tensor=True)
#     embeddings2 = model.encode(text2, convert_to_tensor=True)

#     # Calculate cosine similarity between the embeddings
#     cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

#     # Convert the similarity tensor to a Python float value
#     similarity_score = cosine_similarity.item()

#     return similarity_score

# # Calculate semantic similarity and store the results in a new column
# data['semantic_similarity'] = data.apply(lambda row: calculate_similarity(row['text1'], row['text2']), axis=1)

# # Save the results to a new CSV file
# data.to_csv('output.csv', index=False)

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify


app = Flask(__name__)


# Load the SentenceTransformer model for sentence embeddings
model = SentenceTransformer('bert-base-nli-mean-tokens')
# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def preprocess_text(text):
    text = text.lower()
    return text

def calculate_similarity(text1, text2):
    # Encode the input texts into sentence embeddings
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)

    # Calculate cosine similarity between the embeddings
    cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

    # Convert the similarity tensor to a Python float value
    similarity_score = cosine_similarity.item()

    return similarity_score

# def calculate_similarity(text1, text2, model, tokenizer, max_length):
#     encoding = tokenizer.encode_plus(
#         text1,
#         text2,
#         add_special_tokens=True,
#         truncation=True,
#         padding='max_length',
#         max_length=max_length,
#         return_tensors='pt'
#     )

#     with torch.no_grad():
#         output = model(**encoding)

#     similarity_score = output.logits[0][0].item()

#     return similarity_score


@app.route('/api/calculate_similarity', methods=['POST'])
def calculate_api():
    data = request.get_json()
    text1 = preprocess_text(data['text1'])
    text2 = preprocess_text(data['text2'])
    max_length = 128

    similarity_score = calculate_similarity(text1, text2)

    response = {
        'similarity_score': similarity_score
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)