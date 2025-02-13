import numpy as np
from agi import AGI

class MemoryManager:
    def __init__(self):
        self.memory_bank = {}

    def get_embedding(self, response):
        embedding = np.array(response['data'][0]['embedding']).flatten()
        print(f"Original embedding shape: {embedding.shape}")

        # Ensure fixed size embeddings using aggregation (e.g., averaging)
        if len(embedding) != 512:
            if len(embedding) > 512:
                # Truncate the embedding
                embedding = embedding[:512]
            else:
                # Pad the embedding with zeros
                embedding = np.pad(embedding, (0, 512 - len(embedding)), 'constant')
            print(f"Adjusted embedding shape: {embedding.shape}")
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm == 0:
            normalized_embedding = embedding
        else:
            normalized_embedding = embedding / norm
        return normalized_embedding

    def add_memory(self, key, text):
        response = self.model.create_embedding(text)
        embedding = self.get_embedding(response)
        self.memory_bank[key] = embedding
        print(f"Memory '{key}' added.")

    def delete_memory(self, key):
        if key in self.memory_bank:
            del self.memory_bank[key]
            print(f"Memory '{key}' deleted.")

    def edit_memory(self, key, new_text):
        if key in self.memory_bank:
            response = self.model.create_embedding(new_text)
            embedding = self.get_embedding(response)
            self.memory_bank[key] = embedding
            print(f"Memory '{key}' edited.")

    def retrieve_memory(self, query):
        response = self.model.create_embedding(query)
        query_embedding = self.get_embedding(response)

        if not self.memory_bank:
            print("Memory bank is empty.")
            return None

        # Compute cosine similarities using dot product (since embeddings are normalized)
        similarities = {key: np.dot(query_embedding, emb) for key, emb in self.memory_bank.items()}
        
        # Debug: Print similarities as percentages
        for key, similarity in similarities.items():
            print(f"{key}: {similarity * 100:.2f}%")

        # Retrieve the most similar memory
        most_relevant = max(similarities, key=similarities.get)
        print(f"Most relevant memory: {most_relevant}")
        return most_relevant

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b)  # Assuming a and b are already normalized