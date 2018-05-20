import os
import numpy as np
import pandas as pd

import settings

def assemble_data():
    files = os.listdir(settings.DATA_DIR)

    for f in files:

        data = pd.read_csv(os.path.join(settings.DATA_DIR, f))
        data = data.drop(settings.DROP_COLS, axis = 1)

        out_filename = os.path.splitext(os.path.basename(os.path.join(settings.DATA_DIR, f)))[0]

        data.to_csv(os.path.join(settings.PROCESSED_DIR, "{}.csv".format(out_filename)))

if __name__ == '__main__':
    assemble_data()

