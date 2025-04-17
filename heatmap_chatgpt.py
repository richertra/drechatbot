import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load the embeddings from the .npy file
embeddings_file_path = 'embeddings3.npy'
embeddings = np.load(embeddings_file_path)

# Calculate the cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Ask the user for orientation preference
orientation = input("Do you want the legend (color bar) to be 'horizontal' or 'vertical'? ").strip().lower()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Determine the orientation of the color bar
if orientation == 'horizontal':
    cbar_orientation = 'horizontal'
    cbar_label_position = 'bottom'
else:
    cbar_orientation = 'vertical'
    cbar_label_position = 'right'

# Create the heatmap with the user-specified orientation
ax = sns.heatmap(similarity_matrix, cmap='coolwarm', annot=False, cbar=True, cbar_kws={"orientation": cbar_orientation})

# Adjust color bar label positioning
if cbar_orientation == 'horizontal':
    ax.figure.axes[-1].xaxis.set_label_position(cbar_label_position)
    ax.figure.axes[-1].set_xlabel('Cosine Similarity')
else:
    ax.figure.axes[-1].yaxis.set_label_position(cbar_label_position)
    ax.figure.axes[-1].set_ylabel('Cosine Similarity')

# Set title and show the plot
plt.title('Cosine Similarity Heatmap')
plt.tight_layout()
plt.show()
