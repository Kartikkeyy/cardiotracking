import os
import logging
from src.config.database import DB_CONNECTION

def run_migrations():
    if not DB_CONNECTION:
        logging.error("Database connection not found, run stopped")
        return

    with DB_CONNECTION.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) UNIQUE NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        DB_CONNECTION.commit()

    migration_dir = os.path.join(os.path.dirname(__file__), '..', 'migrations')
    migration_files = sorted(os.listdir(migration_dir))

    for migration_file in migration_files:
        if not migration_file.endswith('.sql'):
            return

        # Check if the migration has already been applied
        with DB_CONNECTION.cursor() as cursor:
            cursor.execute("SELECT filename FROM migrations WHERE filename = %s", (migration_file,))
            if cursor.fetchone():
                print(f"Skipping already applied migration: {migration_file}")
                continue

        print(f"Applying migration: {migration_file}")
        with open(os.path.join(migration_dir, migration_file), 'r') as f:
            sql_commands = f.read()

        # Execute the SQL commands
        with DB_CONNECTION.cursor() as cursor:
            cursor.execute(sql_commands)
            cursor.execute("INSERT INTO migrations (filename) VALUES (%s)", (migration_file,))
        DB_CONNECTION.commit()

    print("All migrations applied successfully.")


if __name__ == "__main__":
    run_migrations()
