import re
from Bio import SeqIO
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt


sample_id=[]
sequence=[]
fold_type=[]
seq = ('train/astral_train.fa')
for seq_record in SeqIO.parse(seq, "fasta"):
    #print("description:",seq_record.description)
    sample_id.append(str(seq_record.id))
    num = 0
    for i in range(8,20):
        if seq_record.description[i] =='.':
            num += 1
        if num == 2:
            #print(seq_record.description[8:i])
            fold_type.append(str(seq_record.description[8:i])) 
            break
    sequence.append(str(seq_record.seq))


#print(sequence)
train = pd.DataFrame(data ={'sample_id':sample_id,'fold_type':fold_type,'sequence':sequence})
#train.to_csv("train_data.csv", sep='\t',index=False)


