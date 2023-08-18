import spacy
import numpy as np
import os

class elmo_processing:

    def __init__(self, model_path='./model'):
        self.model = spacy.load('en_core_web_sm')
        print(self.model)

    def embedding(self, text_list):
        embeddings_list = [self.model(text).vector.tolist() for text in text_list]
        # embeddings_list = embedding.tolist()
        response_embedding = self.transform_embedding_to_dict(embeddings_list,text_list)
        return response_embedding

    def transform_embedding_to_dict(self, embedding_list, text_list, model_name="text-embedding-elmo-002"):
        prompt_tokens = sum(len(text) for text in text_list)
        total_tokens = sum(len(embedding) for embedding in embedding_list)

        transformed_data = {
            "data": [
                {
                    "embedding": embedding,
                    "index": index,
                    "object": "embedding"
                }
                for index, embedding in enumerate(embedding_list)
            ],
            "model": model_name,
            "object": "list",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens
            }
        }
        return transformed_data