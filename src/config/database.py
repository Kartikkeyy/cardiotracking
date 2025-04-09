import psycopg2
import os
import atexit
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')


class DB:
    def __init__(self):
        self.connection = None

    def connect(self):
        self.connection = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )

    def close(self):
        self.connection.close()


db_instance = DB()
db_instance.connect()

DB_CONNECTION = db_instance.connection

atexit.register(db_instance.close)
