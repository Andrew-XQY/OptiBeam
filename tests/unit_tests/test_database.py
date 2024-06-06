from conftest import *

DATABASE_ROOT = '../../ResultsCenter/db/liverpool.db'

db = database.SQLiteDB(DATABASE_ROOT)
schema={"image_id":"","":"","":"","":"","":"","":"","":"","":""}

db.create_table(table_name="mmf_dataset_metadata", schema={})
