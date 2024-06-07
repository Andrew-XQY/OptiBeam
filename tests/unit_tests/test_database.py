from conftest import *

DATABASE_ROOT = '../../ResultsCenter/db/liverpool.db'

db = database.SQLiteDB(DATABASE_ROOT)
schema={"image_id":"INTEGER PRIMARY KEY",
        "capture_time":"TEXT",
        "original_crop_pos":"TEXT",
        "speckle_crop_pos":"TEXT",
        "beam_parameters":"TEXT",
        "is_sim_added":"BOOLEAN DEFAULT FALSE",
        "experiment_config_id":"INTEGER",
        "comments":"TEXT"}

db.create_table(table_name="mmf_dataset_metadata", schema={})
