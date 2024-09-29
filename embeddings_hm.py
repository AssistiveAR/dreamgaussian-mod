"""
Load the pre-trained embeddings from the file and print the shape of the embeddings.
"""

import pickle


def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# Input Embeddings
Embeddings = load_embeddings('./embeds/text_embedding.pkl')

print("Embeddings shape:")
for key, tensor in Embeddings.items():
    if hasattr(tensor, 'shape'):
        print(f"## {key}: shape = {tensor.shape}")
    else:
        print(f"## {key}: Not a tensor or no shape attribute")
