import os
import openai
import psycopg2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
import time
import random

from collections import Counter
import itertools


# Set OpenAI API Key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Connect to the original database using environment variables
logging.info('Connecting to source data...')
conn_original = psycopg2.connect(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASS'),
    host=os.getenv('DB_HOST'),
    port=int(os.getenv('DB_PORT'))
)

# Connect to the processed database using environment variables
logging.info('Connecting to processed db...')
conn_processed = psycopg2.connect(
    dbname=os.getenv('DB2_NAME'),
    user=os.getenv('DB2_USER'),
    password=os.getenv('DB2_PASS'),
    host=os.getenv('DB2_HOST'),
    port=int(os.getenv('DB2_PORT'))
)

logging.info('Successfully connected to both databases!')

def fetch_data():
    """Fetch descriptions and genres from the original database."""
    logging.info('Fetching data from original database...')
    cursor = conn_original.cursor()
    cursor.execute("SELECT id, description, genre FROM media_items;")
    data = cursor.fetchall()
    # inspect the data to ensure we're getting a tuple
    logging.info(f"First 3 rows of data: {data[:3]}")
    cursor.close()
    logging.info(f"Fetched {len(data)} records.")
    return data

# extract the keywords by cluster dynamically

def extract_keywords_by_cluster(data, labels):
    """Extract the most common genre keywords for each cluster."""
    logging.info("Extracting keywords for each cluster...")
    cluster_keywords = {}

    # Group genres by cluster
    genres_by_cluster = {}
    for row, label in zip(data, labels):
        _, _, genre = row  # Extract genres from data _, ignores the column
        if label not in genres_by_cluster:
            genres_by_cluster[label] = [] # create a new array with this label
        genres_by_cluster[label].append(genre) #append the genre

    # Process keywords for each cluster
    for cluster_label, genres in genres_by_cluster.items():
        # Split comma-separated genres into individual words
        all_keywords = list(itertools.chain(*[g.split(", ") for g in genres]))

        # Count most common keywords
        keyword_counts = Counter(all_keywords)
        top_keywords = ", ".join([f"{word} ({count})" for word, count in keyword_counts.most_common(3)])

        cluster_keywords[cluster_label] = top_keywords
        logging.info(f"Cluster {cluster_label}: {top_keywords}")

    return cluster_keywords





# vectorize the merged data using openai - batching.. running on a t3.micro.
# batching code by gpt4o ;)
def vectorize_data(data, batch_size=10):
    """ Vectorize the merged data using OpenAI's API. """
    factors = [f"{description}, Genre: {genre}" for _, description, genre in data]  # Combine description and genre
    logging.info('Vectorizing merged data...')
    embeddings = []
    for i in range(0, len(factors), batch_size):
        batch = factors[i:i + batch_size]
        try:
            response = openai.Embedding.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            batch_embeddings = [r['embedding'] for r in response['data']]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise
    return np.array(embeddings)


# scale & fit transform the embeddings
def scale_embeddings(embeddings):
    """Standardize the embeddings."""
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    return scaled_embeddings

# apply principal component analysis for dimensionality reduction
def apply_pca(embeddings, n_components=2):
    """Apply PCA for dimensionality reduction. override n in main"""
    # instantiate
    pca = PCA(n_components=n_components)
    # fit transform embeddings & return
    reduced_embeddings = pca.fit_transform(embeddings)
    # stash variance ratio
    info = f"PCA applied. Explained variance ratio: {pca.explained_variance_ratio_}"
    # log it
    logging.info(info)
    # review the first 5 rows of list data
    logging.info(f"First 5 rows of reduced embeddings: {reduced_embeddings[:5]}")
    return reduced_embeddings

def cluster_data(vectors, n_clusters=5):
    """Cluster the embeddings using KMeans."""
    logging.info('Clustering data...')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(vectors)
    score = silhouette_score(vectors, labels)
    logging.info(f"Silhouette Score: {score}")
    return labels, kmeans.cluster_centers_, score


def save_clusters_to_db(data, labels, original_embeddings, reduced_embeddings, cluster_centers):
    """Save clustered data and centroids to the processed database."""
    logging.info('Saving clustered data to processed database...')
    logging.info(f"Data length: {len(data)}")
    logging.info(f"Labels length: {len(labels)}")
    logging.info(f"Original embeddings length: {len(original_embeddings)}")
    logging.info(f"Reduced embeddings length: {len(reduced_embeddings)}")

    try:
        conn_processed = psycopg2.connect(
            dbname=os.getenv('DB2_NAME'),
            user=os.getenv('DB2_USER'),
            password=os.getenv('DB2_PASS'),
            host=os.getenv('DB2_HOST'),
            port=int(os.getenv('DB2_PORT'))
        )
        cursor = conn_processed.cursor()

        # Create tables if they don't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clustered_media_items (
                id SERIAL PRIMARY KEY,
                original_id INT,
                description TEXT,
                genre TEXT,
                cluster_label INT,
                embedding VECTOR(1536),
                reduced_embedding VECTOR(2)
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clustered_centroids (
                id SERIAL PRIMARY KEY,
                centroid VECTOR(2)
            );
        """)

        # Insert clustered data
        for row, label, embedding, reduced_embedding in zip(data, labels, original_embeddings, reduced_embeddings):
            logging.info(f"Data: {data[:5]}")
            logging.info(f"Labels: {labels[:5]}")
            logging.info(f"Original embeddings: {original_embeddings[:5]}")
            logging.info(f"Reduced embeddings: {reduced_embeddings[:5]}")
            try:
                original_id, description, genre = row
                cursor.execute(
                    """
                    INSERT INTO clustered_media_items (original_id, description, genre, cluster_label, embedding, reduced_embedding)
                    VALUES (%s, %s, %s, %s, %s, %s);
                    """,
                    (original_id, description, genre, int(label), embedding.tolist(), reduced_embedding.tolist())
                )
            except Exception as e:
                logging.error(f"Failed to insert record: {row}. Error: {e}")
                continue

        # Insert cluster centroids
        for center in cluster_centers:
            cursor.execute(
                """
                INSERT INTO clustered_centroids (centroid)
                VALUES (%s);
                """,
                (center.tolist(),)
            )

        # Commit the transaction
        conn_processed.commit()
        logging.info("Clustered data and centroids saved successfully.")

    except psycopg2.Error as db_error:
        logging.error(f"An error occurred while saving data: {db_error}")
        conn_processed.rollback()

    finally:
        cursor.close()
        conn_processed.close()

def main():
    """Main execution pipeline."""
    try:
        # 1 get data
        data = fetch_data()
        # 3 embed data
        original_embeddings = vectorize_data(data)
        # 4 scale data
        scaled_embeddings = scale_embeddings(original_embeddings)
        # 5 apply PCA
        reduced_embeddings = apply_pca(scaled_embeddings, n_components=2)
        # 6 use Kmeans to cluster data with reduced embeddings
        labels, cluster_centers, score = cluster_data(reduced_embeddings)
        # 7 extract keywords per cluster
        cluster_keywords = extract_keywords_by_cluster(data, labels)
        # Log keywords for review
        for label, keywords in cluster_keywords.items():
            logging.info(f"Cluster {label}: {keywords}")
        # 8 save into the database
        save_clusters_to_db(data, labels, original_embeddings, reduced_embeddings, cluster_centers)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        conn_original.close()
        conn_processed.close()
        logging.info('Connections to databases closed.')


if __name__ == "__main__":
    main()