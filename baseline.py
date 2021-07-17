import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score
import gc
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置显卡
#print(torch.cuda.is_available())
""" RANDOM_SEED = 42
np.random.seed(RANDOM_SEED) """

# 训练集9472条数据,0~244一共245个标签
train = pd.read_csv('train/train_data.csv', sep='\t')
#print(train)
#print(train['fold_type'].value_counts())

""" sns.countplot(x=train.fold_type)
plt.xlabel('fold_type count')
plt.show() """

# 将fold_type转换为id（0，1，2，...）
train_label_id2type=dict(enumerate(train.fold_type.unique()))
train_label_type2id={value: key for key,value in train_label_id2type.items()}
#print(train_label_type2id)
train['label']=train['fold_type'].map(train_label_type2id)
train = train.sample(frac=1) #打乱数据
train=train[['sequence','label']]
#print(train)

train_df = train[['sequence', 'label']][:8000]
eval_df = train[['sequence', 'label']][8000:]
#print(train_df)
