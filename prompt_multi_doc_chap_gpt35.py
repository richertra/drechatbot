import pandas as pd
import openai
import os
import re  
import time
from datetime import datetime, timedelta

#Ce script utilise l'API GPT-3.5 de OpenAI pour répondre à des questions à choix multiples à partir d'un fichier Excel contenant les questions et les choix possibles. 
#Il prend également en compte un contexte spécifique fourni dans un fichier texte. Voici les points d'entrée du script :
#    read_context_file(file_path) : Cette fonction lit le contenu d'un fichier texte donné et le renvoie sous forme de chaîne de caractères.
#    generate_text(question, choices, context) : Cette fonction génère une réponse à une question à choix multiples en utilisant l'API GPT-3.5 de OpenAI. Elle prend en compte le contexte fourni et retourne la réponse générée par le modèle.
#    process_questions(file_path, context) : Cette fonction traite les questions à choix multiples à partir d'un fichier Excel donné. Elle utilise la fonction generate_text pour obtenir les réponses aux questions et gère également l'utilisation des tokens API pour éviter les limites de débit.
#    extract_answer_letter(raw_response) : Cette fonction extrait les lettres des réponses correctes à partir de la réponse brute générée par le modèle.
#Le script prend ensuite les résultats, les compare aux réponses correctes du fichier Excel d'origine, calcule l'exactitude de chaque réponse et génère un nouveau fichier Excel contenant les résultats ainsi que l'exactitude globale.


def estimate_tokens(prompt):
    return len(prompt.split())  # Remarque: cette estimation est basique.

def read_context_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

LIMIT_TPM = 60000
THRESHOLD_PERCENTAGE = 0.8  # 80% de la limite de tokens
used_tokens = 0
last_reset_time = datetime.now()

def reset_token_usage():
    global used_tokens, last_reset_time
    if datetime.now() - last_reset_time >= timedelta(minutes=1):
        used_tokens = 0
        last_reset_time = datetime.now()

def generate_text(question, choices, context):
    global used_tokens
    openai.api_key = os.environ.get("OPENAI_API_KEY2")
    letters = ['A', 'B', 'C', 'D', 'E']
    formatted_choices = '\n'.join([f"{letters[i]}. {choice}" for i, choice in enumerate(choices)])
    prompt = f"Contexte suivant est fourni pour aider à répondre à la question:\n{context}\n\nQuestion: {question}\n{formatted_choices}\n\nChoisir la ou les réponses correctes identifiées de A à E en indiquant pour chaque lettre faux ou vrai par exemple A: faux ou A: vrai, ainsi de suite pour chaque lettre"

    estimated_tokens = estimate_tokens(prompt)
    if estimated_tokens > LIMIT_TPM * THRESHOLD_PERCENTAGE:
        print(f"Estimated tokens ({estimated_tokens}) exceed {int(THRESHOLD_PERCENTAGE * 100)}% of the limit.")
        return None  # Pause handled in the process_questions function

    reset_token_usage()

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            #model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "Je vais choisir la ou les réponses correctes parmi les choix donnés, en tenant compte du contexte fourni."},
                {"role": "user", "content": prompt}
            ]
        )
        response_tokens = len(response['choices'][0].message['content'].split())
        used_tokens += response_tokens
        return response.choices[0].message["content"].strip()
    except openai.error.RateLimitError as e:
        print(f"Rate limit error: {e}. Pausing for a bit...")
        time.sleep(60)
        return generate_text(question, choices, context)

def process_questions(file_path, context):
    questions_df = pd.read_excel(file_path)
    results = []
    question_count = 0
    
    for index, row in questions_df.iterrows():
        question = row.get('Question')
        if pd.isna(question):
            continue

        choices = [row[f'Answer {i}'] for i in range(1, 6) if f'Answer {i}' in row and not pd.isna(row[f'Answer {i}'])]
        if not choices:
            continue

        answer = generate_text(question, choices, context)
        if answer:
            results.append((question, answer))
            question_count += 1
        
        # Pause after every 4 questions to manage token usage
        if question_count % 4 == 0:
            print(f"Processed {question_count} questions, pausing to manage API token usage...")
            time.sleep(70)  # Adjust time as necessary

    return results

# Remaining parts of the script for extracting letters and calculating accuracy...


def extract_answer_letter(raw_response):
    # Trouver toutes les instances de "Lettre: vrai"
    correct_letters = re.findall(r"([A-E]): vrai", raw_response)
    # Si correct_letters est vide et raw_response n'est pas vide, renvoyer raw_response
    if not correct_letters and raw_response:
        return raw_response  # Renvoie la réponse brute si aucun format attendu n'est trouvé
    return ', '.join(correct_letters)

# Chemin vers le fichier Excel et le fichier de contexte
excel_file = #ADAPTER EN FONCTION 
context_file = #ADAPTER EN FONCTION 


# Lire le contexte du fichier texte
context = read_context_file(context_file)

# Traitement des questions et récupération des résultats avec le contexte
results = process_questions(excel_file, context)

# Création du DataFrame pour les résultats
results_df = pd.DataFrame(results, columns=["Question", "Réponse brute du modèle"])

# Ajout de la colonne "Réponse correcte" issue du fichier original
original_df = pd.read_excel(excel_file)
results_df["Réponse correcte"] = original_df["Answer"]

def calculate_accuracy(row):
    correct_answers = str(row["Réponse correcte"]).split(",")
    deduced_answer = str(row["Lettre de réponse déduite de la réponse brute"])
    return deduced_answer in correct_answers

# Ajouter une colonne pour la "Lettre de réponse déduite de la réponse brute"
results_df["Lettre de réponse déduite de la réponse brute"] = results_df["Réponse brute du modèle"].apply(extract_answer_letter)

# Calcul de l'accuracy pour chaque question
results_df["Accuracy"] = results_df.apply(calculate_accuracy, axis=1)

# Calcul de l'accuracy globale
accuracy_globale = results_df["Accuracy"].mean()

# DataFrame pour l'accuracy globale
accuracy_row = pd.DataFrame([["Accuracy globale", "", "", "", accuracy_globale]], columns=["Question", "Réponse brute du modèle", "Lettre de la réponse extraite", "Réponse correcte", "Accuracy"])

# Concaténation des DataFrames
results_df = pd.concat([results_df, accuracy_row], ignore_index=True)

# Enregistrement
output_excel = 'C:/Users/rrichert/Documents/Travail/Travail avec faculté dentaire/articles/Entrainer modele LLM Endo/test python/GPT4/results_doc_chap.xlsx'
results_df.to_excel(output_excel, index=False)
