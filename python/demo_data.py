from pmll.data import Data, DataReader

filename = 'pmll/dataset/bread.tsv'
with open(filename) as stream:
    dr = DataReader()
    data = dr.read(stream)

print data
