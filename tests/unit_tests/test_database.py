from conftest import *

DATABASE_ROOT = "../../ResultsCenter/db/liverpool.db"

db = database.SQLiteDB(DATABASE_ROOT)
schema = {
            "id":"INTEGER PRIMARY KEY AUTOINCREMENT",
            "image_id":"TEXT",
            "capture_time":"TEXT",
            "original_crop_pos":"TEXT",
            "speckle_crop_pos":"TEXT",
            "beam_parameters":"TEXT",
            "is_sim_added":"BOOLEAN DEFAULT FALSE",
            "config_id":"INTEGER",
            "comments":"TEXT",
            "create_time":"TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "update_time":"TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "is_deleted":"BOOLEAN DEFAULT FALSE"
         }

# create the main table
table_name = "mmf_dataset_metadata"
db.create_table(table_name=table_name, schema=schema)
sql = f"""
            CREATE TRIGGER IF NOT EXISTS update_{table_name}_time
            AFTER UPDATE ON {table_name}
            FOR EACH ROW
            BEGIN
                UPDATE {table_name} SET update_time = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
        """
db.sql_execute(sql) # create the trigger


schema = {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "experiment_name":"TEXT",
            "experiment_description":"TEXT",
            "experiment_location":"TEXT",
            "fiber_config":"TEXT",
            "camera_config":"TEXT",
            "other_config":"TEXT",
            "hash":"TEXT",
            "create_time":"TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "update_time":"TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
}
# create the config table
table_name = "mmf_dataset_config"
db.create_table(table_name=table_name, schema=schema)
sql = f"""
            CREATE TRIGGER IF NOT EXISTS update_{table_name}_time
            AFTER UPDATE ON {table_name}
            FOR EACH ROW
            BEGIN
                UPDATE {table_name} SET update_time = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
        """
db.sql_execute(sql) # create the trigger
