"""
script to create the database and tables based on the schema for fiber speckle dataset.
And other initialization steps.
"""
from conf import *

db = database.SQLiteDB(DATABASE_ROOT)

schema = {
            "id":"INTEGER PRIMARY KEY AUTOINCREMENT", # primary key
            "image_id":"TEXT",  # full timestamp when the image was captured
            "capture_time":"TEXT",  # date and time when the image was captured, to second level
            "original_crop_pos":"TEXT",  # two points for cropping the image
            "speckle_crop_pos":"TEXT",  # two points for cropping the image
            "beam_parameters":"TEXT",  # beam parameters used in the experiment (JSON or dict)
            "num_of_images":"INTEGER",  # number of individual images in the data sample
            "image_path":"TEXT",  # path to the image file
            "metadata_id":"INTEGER",  # foreign key to the config table
            "comments":"TEXT",
            "batch":"INTEGER",  # batch number for the data sample collected in a single run
            "create_time":"TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "update_time":"TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "is_deleted":"BOOLEAN DEFAULT FALSE"
         }  # schema for the main experimental metadata table

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
db.sql_execute(sql) # create the trigger for update_time field


schema = {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "experiment_name":"TEXT",
            "experiment_description":"TEXT",
            "experiment_location":"TEXT",
            "fiber_config":"TEXT",  # JSON or dict
            "camera_config":"TEXT",  # JSON or dict
            "other_config":"TEXT",  # JSON or dict
            "hash":"TEXT",  # hash of the total config for fast comparison
            "create_time":"TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "update_time":"TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
}
# create the config table that stores the detailed setup, parameters and configurations of the experiment
# id is corresponding to the config_id in the main table as a foreign key
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
db.close()
