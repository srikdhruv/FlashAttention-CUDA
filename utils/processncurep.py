import pandas as pd
from tabulate import tabulate

rep_path = './report01.csv'

df = pd.read_csv(rep_path, index_col=0)
print(tabulate(df, headers='keys', tablefmt='pretty'))
