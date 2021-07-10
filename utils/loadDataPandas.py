import pandas as pd

def read(path, column_names):
  dataset = pd.read_csv(
    path,
    na_values='?',
    comment='\t',
    sep=',',
    skipinitialspace=True,
    encoding='cp949'
  )
  return dataset.copy()[column_names]
