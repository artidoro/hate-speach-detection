#%%
import pandas
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

#%%
dev_data_path = '/home/artidoro/data/civility_data/dev.tsv'
mini_demographic_data_path = '/home/artidoro/data/civility_data/mini_demographic_dev.tsv'

#%%
dev_data = pandas.read_csv(dev_data_path, sep='\t')
mini_demographic = pandas.read_csv(mini_demographic_data_path, sep='\t')

#%%
def accuracy(df):
    y_p = df['perspective_score'] > 0.8
    if 'label' in df:
        y_t = df['label'] == 'OFF'
    else:
        y_t = np.zeros(len(y_p), dtype=bool)
    return np.average(y_p == y_t)

def f1(df):
    y_p = df['perspective_score'] < 0.8
    if 'label' in df:
        y_t = df['label'] == 'NOT'
    else:
        y_t = np.ones(len(y_p), dtype=bool)
    return f1_score(y_t, y_p)

def conf_matrix(df):
    y_p = df['perspective_score'] > 0.8
    if 'label' in df:
        y_t = df['label'] == 'OFF'
    else:
        y_t = np.zeros(len(y_p), dtype=bool)
    return confusion_matrix(y_t, y_p)

def fpr(df):
    cm = conf_matrix(df)
    fp = cm[0,1]
    tn = cm[0,0]
    return fp / (fp + tn)


#%%
y_pred = dev_data['perspective_score'] > 0.8
y_true = dev_data['label'] == 'OFF'

#%%
# Accuracy Score
print('Accuracy dev: ', accuracy(dev_data))
print('Accuracy mini_demographic: ', accuracy(mini_demographic))
#%%
# F1 score
print('F1 dev: ', f1(dev_data))
print('F1 mini_demographic: ', f1(mini_demographic))

# %%
# Confusion Matrix
print('CM dev: ', conf_matrix(dev_data))
print('FPR dev: ', fpr(dev_data))
print('CM mini_demographic: ', conf_matrix(mini_demographic))
print('FPR mini_demographic: ', fpr(mini_demographic))

# %%
demographics = set(mini_demographic['demographic'])
for dem in demographics:
    dem_df = mini_demographic[mini_demographic['demographic'] == dem]
    print(dem, ' FPR: ', fpr(dem_df))