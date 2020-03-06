#%%
import pandas
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

#%%
train_data_path = '/home/artidoro/data/civility_data/train.tsv'
dev_data_path = '/home/artidoro/data/civility_data/dev.tsv'
mini_demographic_data_path = '/home/artidoro/data/civility_data/mini_demographic_dev.tsv'

#%%
train_data = pandas.read_csv(train_data_path, sep='\t')
dev_data = pandas.read_csv(dev_data_path, sep='\t')
mini_demographic = pandas.read_csv(mini_demographic_data_path, sep='\t')
mini_demographic['label'] = ['NOT'] * len(mini_demographic['text'])
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

def statistics(df):
    print('Accuracy: ', accuracy(df))
    print('F1: ', f1(df))
    print('FPR: ', fpr(df))
    print('CM: ', conf_matrix(df))

def statistics_predictions(y_p, y_t):
    print('Accuracy: ', np.average(y_p == y_t))
    print('F1: ', f1_score(y_t, y_p, pos_label='NOT'))
    cm = confusion_matrix(y_t, y_p)
    fp = cm[0,1]
    tn = cm[0,0]
    print('FPR: ', fp / (fp + tn))


def load_predictions(path, header=None):
    df = pandas.read_csv(path, header=header)
    return df


#%%
statistics(dev_data)
#%%
predictions_dev = load_predictions('log/cbow_eval/cbow_dev_results.txt')
statistics_predictions(predictions_dev[0], dev_data['label'])
#%%
predictions_md = load_predictions('log/cbow_eval/cbow_mini_demographics_results.txt')
predictions_clean = pandas.DataFrame()
predictions_clean[0] = [pred if pred in ['NOT', 'OFF'] else 'NOT' for pred in predictions_md[0]]
statistics_predictions(predictions_clean[0], mini_demographic['label'])

#%%
demographics = set(mini_demographic['demographic'])
new_data_frame = mini_demographic.copy()
new_data_frame['prediction_cbow'] = predictions_clean[0]
for dem in demographics:
    dem_df = new_data_frame[mini_demographic['demographic'] == dem]
    cm = confusion_matrix(dem_df['label'], dem_df['prediction_cbow'])
    fp = cm[0,1]
    tn = cm[0,0]
    print(dem, ' FPR: ', fp / (fp + tn))


#%%
#CNN
predictions_dev = load_predictions('log/cnn_subword/cnn_dev_results.txt')
statistics_predictions(predictions_dev[0], dev_data['label'])
#%%
predictions_md = load_predictions('log/cnn_subword/cnn_mini_demographic_results.txt')
predictions_clean = pandas.DataFrame()
predictions_clean[0] = [pred if pred in ['NOT', 'OFF'] else 'NOT' for pred in predictions_md[0]]
statistics_predictions(predictions_clean[0], mini_demographic['label'])

#%%
demographics = set(mini_demographic['demographic'])
new_data_frame = mini_demographic.copy()
new_data_frame['prediction_cnn'] = predictions_clean[0]
#%%
set(new_data_frame['prediction_cnn'])
#%%
for dem in demographics:
    dem_df = new_data_frame[mini_demographic['demographic'] == dem]
    cm = confusion_matrix(dem_df['label'], dem_df['prediction_cnn'])
    if len(cm[0])==1:
        print(dem, ' FPR: ', 0)
    else:
        fp = cm[0,1]
        tn = cm[0,0]
        print(dem, ' FPR: ', fp / (fp + tn))


#%%
with open('/home/artidoro/data/civility_data/test.tsv', 'r') as infile:
    infile.readline()
    test_lines = [line.strip() for line in infile.readlines()]

#%%
test_df_1 = pandas.DataFrame()
test_df_1['text'] = test_lines
#%%
cbow_pred_df = load_predictions('log/cbow_eval/cbow_test_results.txt')

#test_prediction_cbow =



























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
#%%

# %%
demographics = set(mini_demographic['demographic'])
for dem in demographics:
    dem_df = mini_demographic[mini_demographic['demographic'] == dem]
    print(dem, ' FPR: ', fpr(dem_df))


#%%
dev_data_path = '/home/artidoro/data/civility_extra_data/demographic_dev.tsv'
train_data_path = '/home/artidoro/data/civility_extra_data/demographic_train.tsv'
labeled_data_path = '/home/artidoro/data/civility_extra_data/labeled_data.tsv'

#%%
dev_data = pandas.read_csv(dev_data_path, sep='\t')
train_data = pandas.read_csv(train_data_path, sep='\t')
labeled_data =  pandas.read_csv(labeled_data_path, sep='\t')


#%%
labeled_data[labeled_data['class']==0]['tweet']

# %%
train_data


# %%
labeled_data['label'] = ['OFF' if label else 'NOT' for label in labeled_data['class'] < 2]


# %%
labeled_data['label']

# %%
new_data = pandas.DataFrame()

# %%
new_data['text'] = labeled_data['tweet'].append(train_data['text'])
#%%
new_data['label'] = labeled_data['label'].append(train_data['label'])
# %%
new_data

# %%
new_data = new_data.sample(frac=1).reset_index(drop=True)
#%%
new_data = new_data[['text', 'label']]
# %%
dest_path = '/home/artidoro/data/civility_extra_data/labeled_composite_train.tsv'
new_data.to_csv(dest_path, sep='\t', index=False)

# %%
