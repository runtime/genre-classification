import os
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
import termplotlib as tpl
from sklearn.metrics import silhouette_score
import logging
import json

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
        cursor.execute("SELECT reduced_embedding::text, cluster_label FROM clustered_media_items;")
        clustered_data = cursor.fetchall()
        # use json.loads and convert to np.float(32)
        embeddings = np.array([np.array(json.loads(row[0]), dtype=np.float32) for row in clustered_data])
        labels = np.array([row[1] for row in clustered_data])

        # Fetch reduced cluster centers
        logging.info("Fetching cluster centers...")
        cursor.execute("SELECT centroid::text FROM clustered_centroids;")
        centroids_data = cursor.fetchall()
        cluster_centers = np.array([np.array(json.loads(row[0]), dtype=np.float32) for row in centroids_data])

        logging.info(f"Reduced embeddings shape: {embeddings.shape}")
        logging.info(f"Labels shape: {labels.shape}")
        logging.info(f"Cluster centers shape: {cluster_centers.shape}")

        return embeddings, labels, cluster_centers
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

def plot_clusters(data, labels, cluster_centers):
    """Generate and save cluster plot as an image with labels."""
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Centroids')

    # Annotate each centroid with its cluster label
    for i, center in enumerate(cluster_centers):
        plt.text(center[0], center[1], f'Cluster {i}', fontsize=12, color='black', ha='center', va='center')

    plt.title('Cluster Visualization with Labels')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()

    # Save plot
    output_dir = "scripts/outputs"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "cluster_plot_labeled.png")
    plt.savefig(plot_path)
    logging.info(f"Plot saved to {plot_path}")

def main():
    """Main execution pipeline."""
    conn = None
    try:
        conn = connect_to_db()

        # Fetch reduced embeddings, labels, and cluster centers
        embeddings, labels, cluster_centers = fetch_clustered_data(conn)

        # Calculate and log Silhouette score
        calculate_silhouette(embeddings, labels)

        # Generate and save cluster plot
        plot_clusters(embeddings, labels, cluster_centers)
    except Exception as e:
        logging.error(f"An error occurred during the pipeline: {e}")
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
