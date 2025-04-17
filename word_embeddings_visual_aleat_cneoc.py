import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import random
from itertools import combinations

# Chemins vers le fichier Excel et le fichier .npy
excel_file_path =#ADAPTER EN FONCTION pour mettre le fichier excel
npy_file_path = "embeddings4.npy"

# Charger les noms des termes depuis le fichier Excel
def load_terms_from_excel(file_path, column_name="Question"):
    df = pd.read_excel(file_path)
    terms = df[column_name].dropna().str.strip().str.lower().tolist()
    return terms

# Charger les embeddings depuis le fichier .npy
def load_embeddings_from_npy(npy_file_path):
    return np.load(npy_file_path)

# Trouver les 10 voisins les plus proches de chaque terme
def find_nearest_neighbors(embeddings, num_neighbors=10):
    nearest_neighbors = []
    for i, embedding in enumerate(embeddings):
        distances = [(j, cosine(embedding, other_embedding)) for j, other_embedding in enumerate(embeddings) if j != i]
        distances.sort(key=lambda x: x[1])  # Trier par distance croissante
        neighbors = [index for index, _ in distances[:num_neighbors]]  # Garder les indices des plus proches
        nearest_neighbors.append(neighbors)
        if i % 50 == 0:  # Afficher la progression chaque 50 termes
            print(f"Traitement du terme {i+1}/{len(embeddings)} : Voisins les plus proches trouvés.")
    return nearest_neighbors

# Trouver le trio le plus proche parmi les voisins et deux termes éloignés
def find_closest_trio_and_random_far_terms(terms, embeddings, nearest_neighbors):
    min_average_distance = float('inf')
    best_trio_indices = None

    # Parcourir chaque terme pour analyser les trios dans ses voisins les plus proches
    for i, neighbors in enumerate(nearest_neighbors):
        for trio_indices in combinations(neighbors, 3):
            i1, i2, i3 = trio_indices
            # Calculer les distances cosinus entre les trois voisins
            distance_ij = cosine(embeddings[i1], embeddings[i2])
            distance_ik = cosine(embeddings[i1], embeddings[i3])
            distance_jk = cosine(embeddings[i2], embeddings[i3])
            average_distance = (distance_ij + distance_ik + distance_jk) / 3

            # Mettre à jour si on a trouvé un meilleur trio
            if average_distance < min_average_distance:
                min_average_distance = average_distance
                best_trio_indices = trio_indices

        if i % 50 == 0:
            print(f"Analyse des trios de voisins pour le terme {i+1}/{len(nearest_neighbors)} : Meilleur trio mis à jour.")

    # Sélectionner deux indices aléatoires pour des termes éloignés
    far_indices = [i for i in range(len(embeddings)) if i not in best_trio_indices]
    random_far_indices = random.sample(far_indices, 2)

    # Combiner le meilleur trio et les termes éloignés
    selected_indices = list(best_trio_indices) + random_far_indices
    selected_terms = [terms[i] for i in selected_indices]
    selected_embeddings = embeddings[selected_indices]

    return selected_terms, selected_embeddings

# Charger les termes et les embeddings
terms = load_terms_from_excel(excel_file_path)
embeddings = load_embeddings_from_npy(npy_file_path)

# Trouver les voisins les plus proches
nearest_neighbors = find_nearest_neighbors(embeddings)

# Trouver le trio le plus proche parmi les voisins et deux mots éloignés
selected_terms, selected_embeddings = find_closest_trio_and_random_far_terms(terms, embeddings, nearest_neighbors)

# Vérifier si des termes ont été trouvés
if not selected_terms:
    print("Aucun terme sélectionné n'a été trouvé dans le fichier Excel.")
else:
    # Normaliser les embeddings pour une meilleure visualisation
    scaler = StandardScaler()
    selected_embeddings_normalized = scaler.fit_transform(selected_embeddings)

    # Créer la heatmap des embeddings sélectionnés
    fig, ax = plt.subplots(figsize=(12, len(selected_terms) * 0.5))  # Ajuste la hauteur en fonction du nombre de termes
    sns.heatmap(selected_embeddings_normalized, cmap="viridis", cbar_kws={'orientation': 'horizontal'}, ax=ax)

    # Personnalisation du graphique
    ax.set_yticks([i + 0.5 for i in range(len(selected_terms))])
    ax.set_yticklabels(selected_terms, rotation=0)
    plt.title("Visualisation du Trio d'Embeddings le Plus Proche + 2 Termes Éloignés")

    # Afficher le graphique
    plt.show()
