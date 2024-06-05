from conftest import *
import sqlite3

DATABASE_ROOT = '../../ResultsCenter/db/'

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect(DATABASE_ROOT + 'liverpool.db')

# Create a cursor object using the cursor() method
cursor = conn.cursor()

# Create table
cursor.execute('''CREATE TABLE sample_table
               (id INT PRIMARY KEY NOT NULL,
                name TEXT NOT NULL,
                age INT NOT NULL);''')

# Insert data into table
cursor.execute("INSERT INTO sample_table (id, name, age) VALUES (1, 'Alice', 30)")
cursor.execute("INSERT INTO sample_table (id, name, age) VALUES (2, 'Bob', 25)")
cursor.execute("INSERT INTO sample_table (id, name, age) VALUES (3, 'Charlie', 35)")

# Commit the changes to the database
conn.commit()

cursor.execute("SELECT * FROM sample_table")
results = cursor.fetchall()
for row in results:
    print(row)

conn.close()
