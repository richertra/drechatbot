import os
import numpy as np
import pandas as pd
import openai
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi
import warnings
from sklearn.metrics import davies_bouldin_score
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# Set the OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY2")

# Supprimer les avertissements inutiles
warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils._pytree._register_pytree_node is deprecated")

# Définir le chemin du fichier d'embeddings
embeddings_file_path = "embeddings3.npy"

# Lire le fichier Excel
file_path = #ADAPTER EN FONCTION 
df = pd.read_excel(file_path)

# Extraire les termes
terms = df['Term'].dropna().tolist()

# Charger les embeddings depuis le fichier npy
if os.path.exists(embeddings_file_path):
    embeddings = np.load(embeddings_file_path)
    print("Embeddings chargés avec succès.")
else:
    raise FileNotFoundError(f"Le fichier d'embeddings {embeddings_file_path} n'existe pas.")

# Vérification de la cohérence entre le nombre de termes et le nombre d'embeddings
if len(terms) != len(embeddings):
    print(f"Erreur : le nombre de termes ({len(terms)}) ne correspond pas au nombre d'embeddings ({len(embeddings)}).")
    min_length = min(len(terms), len(embeddings))
    terms = terms[:min_length]
    embeddings = embeddings[:min_length]

# Réduction de dimension avec t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=3000)
tsne_results = tsne.fit_transform(embeddings)

# Clustering avec KMeans pour déterminer des clusters
n_clusters = 8  # Choisir le nombre de clusters souhaités
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(tsne_results)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Calculer et afficher l'indice de Davies-Bouldin
db_score = davies_bouldin_score(tsne_results, labels)
print(f"Davies-Bouldin Index: {db_score}")

# Méthode Elbow pour choisir le nombre optimal de clusters
distortions = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(tsne_results)
    distortions.append(kmeans.inertia_)

# Afficher le graphe de la méthode Elbow
plt.figure(figsize=(8, 5))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Nombre de clusters')
plt.ylabel('Distorsion (Inertia)')
plt.title('Méthode Elbow pour trouver le k optimal')
plt.show()

# Création de la partition de Voronoi
vor = Voronoi(centroids)

# Définir les limites du graphique pour recadrer les polygones de Voronoi
x_min, x_max = -125, 90
y_min, y_max = -70, 100

def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Fonction conçue uniquement pour les diagrammes 2D.")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

# Calculer les polygones finis pour Voronoi
regions, vertices = voronoi_finite_polygons_2d(vor)

plt.figure(figsize=(12, 10))

# Utiliser la même colormap que pour les points
cmap = plt.cm.get_cmap('tab10', n_clusters)

# Dessiner les régions de Voronoi et colorier les cellules en fonction du cluster correspondant
for region_index, region in enumerate(regions):
    polygon = vertices[region]
    plt.fill(*zip(*polygon), color=cmap(region_index), alpha=0.3)
    plt.plot(*zip(*polygon), color='black', linewidth=1.5)

# Visualisation des points avec des couleurs pour chaque cluster
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=cmap, edgecolor='k', alpha=0.8)

# Annoter les points avec les termes
for i, term in enumerate(terms):
    plt.annotate(term, (tsne_results[i, 0], tsne_results[i, 1]), fontsize=8, color='black', alpha=0.7,
                 xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

# Ajouter les centres des clusters
plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='white', edgecolors='k', marker='X', linewidths=2)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.show()
