import sqlite3
from typing import *

class SQLiteDB:
    def __init__(self, db_path: str):
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        
    def create_table(self, table_name: str, schema: Dict[str, str], add_base_schema=True) -> None:
        """
        Creates a table with an auto-incrementing ID, created_at, and modified_at fields,
        along with user-defined schema.
        :param table_name: Name of the table to create.
        :param schema: Dictionary of column names and their SQL data types.
        """
        base_schema = {}
        if add_base_schema:
            base_schema = {
                'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',
                'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                'modified_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                'is_deleted': 'BOOLEAN DEFAULT FALSE'
            }
        # Merging user-defined schema with the base schema
        full_schema = {**schema, **base_schema}
        columns = ', '.join(f"{col_name} {data_type}" for col_name, data_type in full_schema.items())
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})")
        self.cursor.execute(f"""
            CREATE TRIGGER IF NOT EXISTS update_{table_name}_modified_time
            AFTER UPDATE ON {table_name}
            FOR EACH ROW
            BEGIN
                UPDATE {table_name} SET modified_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
        """)
        self.connection.commit()
        # Adding a trigger to update the modified_at column on update
        self.create_update_trigger(table_name)

    def add_field(self, table_name: str, column_name: str, data_type: str) -> None:
        """
        Add a new field to an existing table.
        :param table_name: Name of the table.
        :param column_name: Name of the new column.
        :param data_type: Data type of the new column.
        """
        self.cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {data_type}")
        self.connection.commit()

    def delete_table(self, table_name: str) -> None:
        """
        Deletes a table from the database.
        :param table_name: Name of the table to delete.
        """
        self.cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.connection.commit()

    def insert_record(self, table_name: str, record: Dict[str, Any]) -> None:
        """
        Inserts a new record into the specified table.
        :param table_name: Name of the table.
        :param record: Dictionary representing the record to insert.
        """
        columns = ', '.join(record.keys())
        placeholders = ', '.join('?' * len(record))
        values = tuple(record.values())
        self.cursor.execute(f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})", values)
        self.connection.commit()

    def delete_record(self, table_name: str, key_column: str, key_value: Any) -> None:
        """
        Deletes a single record from a table based on the key column and key value.
        :param table_name: Name of the table.
        :param key_column: Name of the key column to match for deletion.
        :param key_value: Value of the key to match for deletion.
        """
        self.cursor.execute(f"DELETE FROM {table_name} WHERE {key_column} = ?", (key_value,))
        self.connection.commit()

    def __end__(self):
        self.cursor.close()
        self.connection.close()
