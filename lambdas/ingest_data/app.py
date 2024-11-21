import os
import psycopg2
import json

def handler(event, context):
    # Connect to Postgres
    conn = psycopg2.connect(
        dbname=os.environ['DB_NAME'],
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'],
        host=os.environ['DB_HOST'],
        port=os.environ['DB_PORT']
    )
    cursor = conn.cursor()

    try:
        # Loop through the event records (assuming multiple inserts)
        for record in event['Records']:
            body = json.loads(record['body'])  # If invoked via SQS, SNS, etc.
            title = body['title']
            media_type = body['type']
            description = body['description']
            thumbnail_url = body['thumbnail_url']
            release_date = body['release_date']

            # Insert into the media_items table
            cursor.execute(
                """
                INSERT INTO media_items (title, type, description, thumbnail_url, release_date)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (title, media_type, description, thumbnail_url, release_date)
            )

        conn.commit()
        return {
            'statusCode': 200,
            'body': 'Data successfully ingested.'
        }
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
        return {
            'statusCode': 500,
            'body': f"Error: {e}"
        }
    finally:
        cursor.close()
        conn.close()
