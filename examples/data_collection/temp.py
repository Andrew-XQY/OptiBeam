from conf import *
import pandas as pd


# ============================
# update specific fields
# ============================
# database_dir = f"../../DataHub/datasets/2024-12-17/db/dataset_meta.db"
# DB = database.SQLiteDB(database_dir)
# sql = """
#     SELECT 
#         id, purpose
#     FROM 
#         mmf_dataset_metadata
#     WHERE 
#         batch = 6 
# """
# df = DB.sql_select(sql)
# df['purpose'] = 'fun'
# sql = DB.batch_update("mmf_dataset_metadata", "id", df)
# print(sql)
# DB.sql_execute(sql, multiple=True)
# DB.close()  



# ============================
# update datasample paths
# ============================
# database_dir = f"../../DataHub/datasets/2024-12-17/db/dataset_meta.db"
# DB = database.SQLiteDB(database_dir)
# sql = """
#     SELECT 
#         id, image_path
#     FROM 
#         mmf_dataset_metadata
# """
# df = DB.sql_select(sql)
# def process_element(x):
#     return x.replace("datasets", "dataset")

# df['image_path'] = df['image_path'].apply(process_element)
# sql = DB.batch_update("mmf_dataset_metadata", "id", df)
# DB.sql_execute(sql, multiple=True)
# DB.close()  



# ============================
# database test
# ============================
database_dir = f"../../DataHub/datasets/2024-12-17/db/dataset_meta.db"
DB = database.SQLiteDB(database_dir)
sql = """
    SELECT 
        id, batch, purpose, image_path, comments
    FROM 
        mmf_dataset_metadata
    WHERE 
        is_calibration = 0 AND purpose = 'testing' AND comments IS NULL
"""
df = DB.sql_select(sql)
print('Total number of records in the table: ' + str(len(df)))
DB.close()  
