import sqlite3
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

    def sql_execute(self, sql: str) -> None:
        """
        Executes a raw SQL command.
        
        args:
            sql (str): SQL command to execute.
            
        returns:
            None
        """
        self.cursor.execute(sql)
        self.connection.commit()
        print(f"SQL command executed")
    
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
        
    def update_record(self, table_name: str, key_column: str, key_value: Any, new_values: Dict[str, Any]) -> None:
        """
        Updates a single record in a table based on the key column and key value.
        
        args:
            table_name (str): Name of the table.
            key_column (str): Name of the key column to match for update.
            key_value (Any): Value of the key to match for update.
            new_values (dict): Dictionary of column names and their new values.
            
        returns:
            None
        """
        set_values = ', '.join(f"{col_name} = ?" for col_name in new_values.keys())
        values = tuple(new_values.values())
        self.cursor.execute(f"UPDATE {table_name} SET {set_values} WHERE {key_column} = ?", values + (key_value,))
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
    
    def batch_update(table_name, id_column, dataframe) -> str:
        """
        Constructs a single SQL statement for batch updating rows in a SQLite table.
        
        Args:
        table_name (str): Name of the database table to update.
        id_column (str): Name of the column that serves as the primary key.
        dataframe (pd.DataFrame): DataFrame containing data to update. 
                                The DataFrame should include the primary key column
                                and other columns that match exactly the column names in the table.
        
        Returns:
        str: A SQL statement for updating multiple rows.
        """

        sql = f"UPDATE {table_name} SET "
        # Generate the SET part of the SQL command dynamically based on DataFrame columns
        columns = dataframe.columns.tolist()
        columns.remove(id_column)  # Remove the ID column from the list to form the SET clause
        set_clause = ', '.join([f"{col} = CASE {id_column} " for col in columns])
        # Generate the CASE part of the SQL for each column
        case_statements = []
        for col in columns:
            cases = ' '.join([f"WHEN {id_column} = {row[id_column]} THEN '{row[col]}'" for index, row in dataframe.iterrows()])
            case_statements.append(f"{cases} END")
        # Combine everything
        set_clause = ', '.join([f"{col} = CASE {id_column} " + case + " END" for col, case in zip(columns, case_statements)])
        sql += set_clause
        # Add WHERE clause to limit the update to only the IDs mentioned in the DataFrame
        ids = ', '.join([str(id) for id in dataframe[id_column].unique()])
        sql += f" WHERE {id_column} IN ({ids})"
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
