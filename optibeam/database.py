import sqlite3
import pandas as pd
from abc import ABC, abstractmethod
from .utils import print_underscore
from typing import *


class Database(ABC):
    def __init__(self):
        self.connection = None
    
    @abstractmethod
    def create_table(self, table_name: str, schema: Dict[str, str]) -> None:
        pass
    
    @abstractmethod
    def delete_table(self, table_name: str) -> None:
        pass

    @abstractmethod
    def add_field(self, table_name: str, column_name: str, data_type: str) -> None:
        pass

    @abstractmethod
    def insert_record(self, table_name: str, record: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def delete_record(self, table_name: str, key_column: str, key_value: Any) -> None:
        pass
    
    @abstractmethod
    def update_record(self, table_name: str, key_column: str, key_value: Any, new_values: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def sql_execute(self, sql: str) -> None:
        pass
    
    @abstractmethod
    def close(self) -> None:
        pass


class SQLiteDB(Database):
    @print_underscore
    def __init__(self, db_path: str):
        """
        Initializes the SQLite database connection and cursor.
        
        args:
            db_path (str): Path to the SQLite database file.
        
        returns:
            None
        """
        super().__init__()
        self.text_types = (str, bytes, set, list, dict, tuple, bool)
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self.tables = self.get_all_tables()
        print(f"{len(self.tables)} table(s) found in the database:")
        for t in sorted(self.tables): print(t)
        
    def get_all_tables(self) -> List[str]:
        """
        Returns a list of all tables in the database.
        
        args:
            None
            
        returns:
            List[str]: List of table names.
        """
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [table[0] for table in self.cursor.fetchall()]
        
    def create_table(self, table_name: str, schema: Dict[str, str]) -> None:
        """
        Creates a new table in the database.
        
        args:
            table_name (str): Name of the table to create.
            schema (dict): Dictionary of column names and their SQL data types.
            
        returns:
            None
        """
        if table_name in self.tables:
            print(f"Table {table_name} already exists.")
            return
        columns = ', '.join(f"{col_name} {data_type}" for col_name, data_type in schema.items())
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})")
        print(f"Table {table_name} created with schema:\n {schema}")

    def sql_execute(self, sql: str, multiple=False) -> None:
        """
        Executes a raw SQL command.
        
        args:
            sql (str): SQL command to execute.
            
        returns:
            None
        """
        if multiple:
            self.cursor.executescript(sql)
        else:
            self.cursor.execute(sql)
        self.connection.commit()
        print(f"SQL command executed")
    
    def sql_select(self, sql: str) -> pd.DataFrame:
        """
        Executes a raw SQL SELECT command.
        
        args:
            sql (str): SQL SELECT command to execute.
            
        returns:
            DataFrame: Pandas DataFrame containing the results of the SELECT command.
        
        raises:
            Exception: If SQL execution fails.
        """
        try:
            return pd.read_sql_query(sql, self.connection)
        except Exception as e:
            print(f"An error occurred: {e}")
            raise
    
    def add_field(self, table_name: str, column_name: str, data_type: str) -> None:
        """
        Add a new field to an existing table.
        
        args:
            table_name (str): Name of the table.
            column_name (str): Name of the new column.
            data_type (str): Data type of the new column.
            
        returns:
            None
        """
        self.cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {data_type}")
        self.connection.commit()
        
    def rename_field(self, table_name: str, old_column_name: str, new_column_name: str) -> None:
        """
        Rename a field in a table.
        
        args:
            table_name (str): Name of the table.
            old_column_name (str): Name of the column to rename.
            new_column_name (str): New name for the column.
            
        returns:
            None
        """
        self.cursor.execute(f"ALTER TABLE {table_name} RENAME COLUMN {old_column_name} TO {new_column_name}")
        self.connection.commit()
        
    def retype_field(self, table_name: str, column_name: str, new_data_type: str) -> None:
        """
        Change the data type of a field in a table.
        
        args:
            table_name (str): Name of the table.
            column_name (str): Name of the column to change.
            new_data_type (str): New data type for the column.
            
        returns:
            None
        """
        self.cursor.execute(f"ALTER TABLE {table_name} ALTER COLUMN {column_name} TYPE {new_data_type}")
        self.connection.commit()

    def delete_table(self, table_name: str) -> None:
        """
        Deletes a table from the database.
        
        args:
            table_name (str): Name of the table to delete.
            
        returns:
            None
        """
        self.cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.connection.commit()
        print(f"Table {table_name} deleted")

    def insert_record(self, table_name: str, record: Dict[str, Any]) -> None:
        """
        Inserts a new record into the specified table.
        
        args:
            table_name (str): Name of the table.
            record (dict): Dictionary representing the record to insert.
            
        returns:
            None
        """
        columns = ', '.join(record.keys())
        placeholders = ', '.join('?' * len(record))
        values = tuple(record.values())
        self.cursor.execute(f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})", values)
        self.connection.commit()

    def delete_record(self, table_name: str, key_column: str, key_value: Any) -> None:
        """
        Deletes a single record from a table based on the key column and key value.
        
        args:
            table_name (str): Name of the table.
            key_column (str): Name of the key column to match for deletion.
            key_value (Any): Value of the key to match for deletion.
            
        returns:
            None
        """
        self.cursor.execute(f"DELETE FROM {table_name} WHERE {key_column} = ?", (key_value,))
        self.connection.commit()
        
    def update_record(self, table_name, key_field, key_value, update_field, update_value) -> None:
        """
        Generate an SQL string and update a single value in an SQLite table.

        args:
            table_name (str): Name of the table.
            key_field (str): Name of the key field.
            key_value (Any): Value of the key field.
            update_field (str): Name of the field to update.
            update_value (Any): Value to update the field to.

        Returns:
            None
        """
        if isinstance(key_value, self.text_types):
            key_value = f"'{key_value}'"
        if isinstance(update_value, self.text_types):
            update_value = f"'{update_value}'"
        
        sql = f"UPDATE {table_name} SET {update_field} = {update_value} WHERE {key_field} = {key_value};"
        self.cursor.execute(sql)
        self.connection.commit()
        
    def get_max(self, table_name, column_name) -> float:
        """
        Get the maximum value of a column in a table, have to be a numeric column.
        
        args:
            table_name (str): Name of the table.
            column_name (str): Name of the column.
            
        returns:
            float: Maximum value in the column.
        """
        query = f"SELECT MAX({column_name}) FROM {table_name}"
        self.cursor.execute(query)
        return self.cursor.fetchone()[0]
    
    def get_min(self, table_name, column_name) -> float:
        """
        Get the minimum value of a column in a table, have to be a numeric column.
        
        args:
            table_name (str): Name of the table.
            column_name (str): Name of the column.
            
        returns:
            float: Minimum value in the column.
        """
        query = f"SELECT MIN({column_name}) FROM {table_name}"
        self.cursor.execute(query)
        return self.cursor.fetchone()[0]
    
    def record_exists(self, table_name, column_name, value) -> bool:        
        """
        Check if an record exists in the table.
        
        args:
            table_name (str): Name of the table.
            column_name (str): Name of the column.
            value (Any): Value to check for existence.
            
        returns:
            bool: True if the record exists, False otherwise.
        """
        query = f"SELECT EXISTS(SELECT 1 FROM {table_name} WHERE {column_name} = ? LIMIT 1)"
        self.cursor.execute(query, (value,))
        exists = self.cursor.fetchone()[0] == 1
        return exists
    
    def batch_update(self, table_name, primary_key, df):
        """
        Generate SQL UPDATE statements for updating rows in an SQLite table based on a DataFrame.

        Args:
        - table_name (str): Name of the SQLite table to be updated.
        - primary_key (str): Column name in the table and DataFrame to use as a primary key.
        - df (pd.DataFrame): DataFrame containing the data to update.

        Returns:
        - str: SQL string that contains all the update commands.
        """
        sql_commands = []
        for _, row in df.iterrows():
            set_clause = ', '.join([f"{col} = '{row[col]}'" if isinstance(row[col], self.text_types) 
                                    else f"{col} = {row[col]}" for col in df.columns if col != primary_key])
            primary_val = f"'{row[primary_key]}'" if isinstance(row[primary_key], self.text_types) else row[primary_key]
            sql_commands.append(f"UPDATE {table_name} SET {set_clause} WHERE {primary_key} = {primary_val};")
        return "\n".join(sql_commands)

    
    def batch_delete(self, table_name, id_column, id_list):
        """
        Construct a SQL DELETE statement for batch deletions based on primary key values.

        Args:
            table_name (str): Name of the table from which to delete rows.
            id_column (str): Column name which is the primary key.
            id_list (list): List of primary key values that indicate rows to be deleted.

        Returns:
            str: SQL DELETE statement.
        """
        # Prepare the parameter placeholders and the SQL statement
        placeholders = ', '.join(['?'] * len(id_list))  # Create a placeholder for each id
        sql = f"DELETE FROM {table_name} WHERE {id_column} IN ({placeholders})"
        return sql
    
    def close(self) -> None:
        """
        Close the database connection.
        
        args:
            None
            
        returns:
            Nones
        """
        self.cursor.close()
        self.connection.close()
        print("Database connection closed")
