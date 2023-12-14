from datasets import load_dataset
import torch
from sentence_transformers.util import semantic_search
import requests
import numpy as np
import ast

import streamlit as st


from huggingface_hub import login
#login(token="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")

def MainQuery(input):

    def get_embedding(text, model_id, hf_token):
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
        headers = {"Authorization": f"Bearer {hf_token}"}
        response = requests.post(api_url, headers=headers, json={"inputs": text, "options":{"wait_for_model":True}})
        return response.json()

    dataset = load_dataset('nBrain-Ai/TestBeddings')

    def parse_embedding(embedding_str):
        return np.array(ast.literal_eval(embedding_str), dtype=np.float32)

    parsed_embeddings = [parse_embedding(embed_str) for embed_str in dataset["train"]["Embedding"]]
    precomputed_embeddings = torch.tensor(np.array(parsed_embeddings), dtype=torch.float32)

    texts = dataset["train"]["Text"]

    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    hf_token = "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"



    query_embedding = torch.FloatTensor(get_embedding(input, model_id, hf_token))

    hits = semantic_search(query_embedding, precomputed_embeddings, top_k=1)

    for hit in hits[0]:
        return texts[hit['corpus_id']]
input = st.text_input("Enter Query Here")

if st.button("Query"):
  st.write(MainQuery(input))
