
import sqlite3
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class spineDataset():
    def __init__(self, database, args):
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.path_to_db = args.path_to_db

        self.db = database
        with self.db.session as sess:
            self.names = [v.pid for v in sess.query(
                self.db.tables.outputs.c.pid).distinct()]
            


    def __len__(self):
        with sqlite3.connect(self.path_to_db, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM images im"
                " INNER JOIN ("
                    f"SELECT o.uuid FROM outputs o"
                    ") o on o.uuid = im.uuid"
                " ORDER BY im.pid"
                )
        all_rows = self.cursor_to_df_fetchall(cursor=cursor)
        return len(all_rows.index)


    def __getitem__(self, index):
        """
        Given an index - searches database for all images matching that patient ID
        """
        pid = self.names[index]
        #* Get patient images of correct modality
        with sqlite3.connect(self.path_to_db, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM images im"
                " INNER JOIN ("
                    f"SELECT o.uuid, o.pid FROM outputs o"
                    f" WHERE o.pid='{pid}'"
                    ") o on o.uuid = im.uuid"
                " ORDER BY im.pid"
                )
        all_rows = self.cursor_to_df_fetchall(cursor=cursor)
        if all_rows.empty:
            logger.error(f"No scans detected for patient: {pid}.") #! Don't think this should ever be triggered?
            return None
        
        return {'images': all_rows}
    

    @staticmethod
    def cursor_to_df_fetchall(cursor):
        #~ Converts database connection to pd.DataFrame
        desc = cursor.description
        column_names = [col[0] for col in desc]
        dict_ = [dict(zip(column_names, row)) for row in cursor]
        return pd.DataFrame(dict_)