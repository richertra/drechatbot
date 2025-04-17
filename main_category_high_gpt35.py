import openai
import pandas as pd
import os
import time

# Remplacez ceci par votre clé d'API GPT-3.5 Turbo
openai.api_key = os.environ.get("OPENAI_API_KEY2")

# Fonction pour générer la réponse de GPT-3.5 Turbo
def generate_category_response(question, categories):
    prompt = f"Given the question: '{question}', what would be the most suitable category from the list of existing categories: {', '.join(categories)}?"
    
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides category suggestions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50  # Limiter le nombre de tokens pour obtenir une réponse concise
    )
    
    return response.choices[0].message["content"].strip()

# Charger le fichier Excel original
excel_file = #ADAPTER EN FONCTION 
original_df = pd.read_excel(excel_file)

# Obtenir les catégories uniques à partir du fichier Excel
categories = original_df['Category'].unique()

# Créer une nouvelle colonne dans le DataFrame pour stocker les réponses de GPT-3.5 Turbo
original_df['GPT-4 Response'] = ""

# Parcourir chaque ligne/question dans le DataFrame en lots de 50
batch_size = 50
for batch_start in range(0, len(original_df), batch_size):
    batch_end = min(batch_start + batch_size, len(original_df))
    batch_df = original_df.iloc[batch_start:batch_end]
    
    for index, row in batch_df.iterrows():
        question = row['Question']
        # Appeler la fonction pour obtenir la réponse de GPT-3.5 Turbo
        response = generate_category_response(question, categories)
        # Enregistrer la réponse dans la colonne appropriée
        original_df.at[index, 'GPT-4 Response'] = response
    
    # Attendre 70 secondes avant de passer au lot suivant
    if batch_end < len(original_df):
        time.sleep(70)

# Fonction pour extraire la catégorie de la réponse générée
def extract_predicted_category(response, categories):
    response = response.lower()  # Convertir en minuscules pour la correspondance insensible à la casse
    for category in categories:
        if category.lower() in response:
            return category  # Catégorie extraite
    return None  # Aucune catégorie extraite

# Appliquer la fonction d'extraction de catégorie et enregistrer dans "Predicted Category"
original_df['Predicted Category'] = original_df['GPT-4 Response'].apply(lambda x: extract_predicted_category(x, categories))

# Calculer l'accuracy globale
original_df['Accuracy'] = original_df.apply(lambda row: row['Predicted Category'] == row['Category'], axis=1)

# Calculer l'accuracy globale
accuracy_globale = original_df['Accuracy'].mean()

# Enregistrer le DataFrame mis à jour dans un nouveau fichier Excel
output_excel = 'category_gpt4_results_with_accuracy_large.xlsx'
original_df.to_excel(output_excel, index=False)

# Imprimer l'accuracy globale
print(f"Accuracy globale : {accuracy_globale}")
