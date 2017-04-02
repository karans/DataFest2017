import pandas as pd

data = pd.read_csv('ASADataFest2017Data/data.txt', sep='\t')
dest = pd.read_csv('ASADataFest2017Data/dest.txt', sep='\t')

# result = pd.merge(data,dest,on='srch_destination_id')
# result.to_jcsv(sep = '\t')