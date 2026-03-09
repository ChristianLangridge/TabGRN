import csv
import pandas as pd

tpm_data = pd.read_fwf('data/Original_TPM_data.txt')
tpm_data.to_csv('Original_TPM_data.csv')