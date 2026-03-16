import GEOparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spatialmt.config import Paths, Dirs, setup_output_dirs, validate_raw_inputs

if __name__ == '__main__':
    setup_output_dirs()
    validate_raw_inputs()

# reading in data
raw_count_csv = pd.read_csv(Paths.raw_count_csv, header=1, index_col=0)

# load the study
gse = GEOparse.get_GEO(geo="GSE153076")

# build GSM mapping 
# looping over to extract title for each column
gsm_to_title = {gsm_id: gsm.metadata['title'][0] for gsm_id, gsm in gse.gsms.items()}

# display the first 10 rows
view_map = pd.Series(gsm_to_title)
print(view_map.head(10))

# rename columns using mapping
raw_count_csv.rename(columns=gsm_to_title, inplace=True)

clean_tpm 