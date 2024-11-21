import os
import openai
import psycopg2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging
import time
import random

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
    cursor.close()
    logging.info(f"Fetched {len(data)} records.")
    return data


# def vectorize_descriptions(descriptions):
#     """Generate embeddings for descriptions using OpenAI."""
#     logging.info('Vectorizing descriptions...')
#     embeddings = []
#
#     # Use the 0.38 version of OpenAI API method for embedding
#     for desc in descriptions:
#         response = openai.Embedding.create(
#             model="text-embedding-ada-002",
#             input=desc
#         )
#         vector = response['data'][0]['embedding']
#         embeddings.append(vector)
#
#     logging.info(f"Generated {len(embeddings)} embeddings.")
#     return np.array(embeddings)
def vectorize_descriptions(descriptions, batch_size=10):
    logging.info('Vectorizing descriptions...')
    embeddings = []
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i + batch_size]
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

def retry_with_backoff(func, *args, retries=5, backoff_factor=1.5, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wait_time = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
            logging.warning(f"Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1}/{retries})")
            time.sleep(wait_time)
    raise RuntimeError(f"Failed after {retries} retries.")

# Use it for vectorization
def vectorize_batch_with_retry(batch):
    return retry_with_backoff(
        openai.Embedding.create,
        input=batch,
        model="text-embedding-ada-002"
    )

def cluster_data(vectors, n_clusters=5):
    """Cluster the embeddings using KMeans."""
    logging.info('Clustering data...')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(vectors)
    score = silhouette_score(vectors, labels)
    logging.info(f"Silhouette Score: {score}")
    return labels, kmeans.cluster_centers_


def save_clusters_to_db(data, labels, cluster_centers):
    """Save clustered data and centroids to the processed database."""
    logging.info('Saving clustered data to processed database...')
    cursor = conn_processed.cursor()

    # Create tables if they don't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clustered_media_items (
            id SERIAL PRIMARY KEY,
            original_id INT,
            description TEXT,
            genre TEXT,
            cluster_label INT,
            embedding VECTOR(1536)
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clustered_centroids (
            id SERIAL PRIMARY KEY,
            centroid VECTOR(1536)
        );
    """)

    # Insert clustered data
    for (original_id, description, genre), label, embedding in zip(data, labels, cluster_centers):
        cursor.execute(
            """
            INSERT INTO clustered_media_items (original_id, description, genre, cluster_label, embedding)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (original_id, description, genre, int(label), embedding.tolist())
        )

    # Insert cluster centroids
    for center in cluster_centers:
        cursor.execute(
            """
            INSERT INTO clustered_centroids (centroid)
            VALUES (%s);
            """,
            (center.tolist(),)
        )

    conn_processed.commit()
    logging.info("Clustered data and centroids saved successfully.")


def main():
    """Main execution pipeline."""
    try:
        data = fetch_data()
        descriptions = [row[1] for row in data]  # Extract descriptions
        vectors = vectorize_descriptions(descriptions)
        labels, cluster_centers = cluster_data(vectors)
        save_clusters_to_db(data, labels, cluster_centers)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        conn_original.close()
        conn_processed.close()
        logging.info('Connections to databases closed.')


if __name__ == "__main__":
    main()
