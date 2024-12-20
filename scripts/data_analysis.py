import os
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import logging
import json
from collections import Counter

logging.basicConfig(level=logging.INFO)

def connect_to_db():
    """Establish a connection to the processed database."""
    try:
        conn_processed = psycopg2.connect(
            dbname=os.getenv('DB2_NAME'),
            user=os.getenv('DB2_USER'),
            password=os.getenv('DB2_PASS'),
            host=os.getenv('DB2_HOST'),
            port=int(os.getenv('DB2_PORT'))
        )
        logging.info("Connected to the processed database.")
        return conn_processed
    except Exception as e:
        logging.error(f"Failed to connect to the database: {e}")
        raise

def fetch_clustered_data(conn):
    """Fetch clustered data and reduced embeddings from the processed database."""
    logging.info("Fetching clustered data...")

    try:
        cursor = conn.cursor()

        # Fetch reduced embeddings and labels
        cursor.execute("SELECT reduced_embedding::text, cluster_label, genre FROM clustered_media_items;")
        clustered_data = cursor.fetchall()
        # us json.loads and convert to np.float data type
        embeddings = np.array([np.array(json.loads(row[0]), dtype=np.float32) for row in clustered_data])
        labels = np.array([row[1] for row in clustered_data])
         # Extract genres
        genres = [row[2] for row in clustered_data]

        # Fetch reduced cluster centers
        logging.info("Fetching cluster centers...")
        # sql statement to get centroids as text from table
        cursor.execute("SELECT centroid::text FROM clustered_centroids;")
        centroids_data = cursor.fetchall()
        cluster_centers = np.array([np.array(json.loads(row[0]), dtype=np.float32) for row in centroids_data])

        logging.info(f"Reduced embeddings shape: {embeddings.shape}")
        logging.info(f"Labels shape: {labels.shape}")
        logging.info(f"Cluster centers shape: {cluster_centers.shape}")

        return embeddings, labels, cluster_centers, genres
    except Exception as e:
        logging.error(f"An error occurred while fetching data: {e}")
        raise
    finally:
        cursor.close()

def calculate_silhouette(data, labels):
    """Calculate and log the silhouette score."""
    score = silhouette_score(data, labels)
    logging.info(f"Silhouette Score: {score}")
    return score

def extract_cluster_keywords(labels, genres):
    """Extract top keywords for each cluster based on genres."""
    logging.info("Extracting keywords for each cluster...")
    cluster_keywords = {}
    for cluster_id in set(labels):
        cluster_genres = [genres[i] for i in range(len(labels)) if labels[i] == cluster_id]
        genre_counts = Counter(", ".join(cluster_genres).split(", "))
        top_keywords = ", ".join([f"{k} ({v})" for k, v in genre_counts.most_common(3)])
        cluster_keywords[cluster_id] = top_keywords
        logging.info(f"Cluster {cluster_id}: {top_keywords}")
    return [cluster_keywords[i] for i in range(len(cluster_keywords))]

def plot_clusters(data, labels, cluster_centers, cluster_labels):
    """Generate and save cluster plot with labels."""
    plt.figure(figsize=(10, 6))

    # Plot individual data points with color coding
    scatter = plt.scatter(
        data[:, 0],
        data[:, 1],
        c=labels,
        cmap='viridis',
        alpha=0.7,
        label='Data Points'  # Add legend for data points
    )

    # Plot cluster centroids
    plt.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        c='red',
        marker='X',
        s=200,
        label='Centroids'  # Add legend for centroids
    )

    # Add labels for centroids
    for idx, label in enumerate(cluster_labels):
        logging.info(f"Adding label '{label}' at {cluster_centers[idx]}")  # Debugging log
        plt.text(
            cluster_centers[idx, 0],
            cluster_centers[idx, 1],
            label,
            fontsize=10,
            ha='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )

    # Add title, labels, and legend
    plt.title('Cluster Visualization with Labels')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(loc='upper right')  # Ensure the legend is displayed

    # Save plot
    output_dir = "scripts/outputs"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "cluster_plot_with_labels.png")
    plt.savefig(plot_path)
    logging.info(f"Plot saved to {plot_path}")


def main():
    """Main execution pipeline."""
    conn = None
    try:
        conn = connect_to_db()

        # Fetch reduced embeddings, labels, and cluster centers
        embeddings, labels, cluster_centers, genres = fetch_clustered_data(conn)

        # Calculate and log Silhouette score
        calculate_silhouette(embeddings, labels)

        # Extract keywords for each cluster
        cluster_labels = extract_cluster_keywords(labels, genres)

        # Generate and save cluster plot
        plot_clusters(embeddings, labels, cluster_centers, cluster_labels)
    except Exception as e:
        logging.error(f"An error occurred during the pipeline: {e}")
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
