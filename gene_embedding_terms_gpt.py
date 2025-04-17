import openai
import pandas as pd
import numpy as np
import os

# Remplacez par votre propre clé API OpenAI
openai.api_key = os.environ.get("OPENAI_API_KEY2")

# Charger le fichier Excel et extraire la colonne "Term"
file_path = #ADAPTER EN FONCTION pour mettre le fichier glossary.xlsx'

# Fonction pour obtenir l'embedding d'un terme en utilisant l'API OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    embedding = response['data'][0]['embedding']
    return embedding

# Charger le fichier Excel et obtenir les embeddings pour chaque terme de la colonne "Term"
def save_embeddings_from_excel(file_path, column_name="Term", output_file="embeddings3.npy"):
    # Charger le fichier Excel
    df = pd.read_excel(file_path)
    
    # S'assurer que la colonne existe
    if column_name not in df.columns:
        raise ValueError(f"La colonne '{column_name}' n'existe pas dans le fichier Excel.")
    
    # Extraire les termes uniques de la colonne "Term"
    terms = df[column_name].dropna().unique()
    
    # Obtenir l'embedding pour chaque terme et les stocker dans une liste
    embeddings = []
    for idx, term in enumerate(terms):
        try:
            embedding = get_embedding(term)
            embeddings.append(embedding)
            print(f"[{idx+1}/{len(terms)}] Embedding pour '{term}' obtenu.")
        except Exception as e:
            print(f"Erreur lors de la création de l'embedding pour '{term}' :", e)
    
    # Convertir la liste des embeddings en un tableau numpy et sauvegarder dans un fichier .npy
    embeddings_array = np.array(embeddings)
    np.save(output_file, embeddings_array)
    print(f"Tous les embeddings ont été sauvegardés dans le fichier : {output_file}")

# Sauvegarder les embeddings de chaque terme dans un seul fichier
save_embeddings_from_excel(file_path)
