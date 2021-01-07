#!/usr/bin/env python
# coding: utf-8

# In[78]:


# Load dataset

import pandas as pd

data_xls = pd.read_excel('Training data.xlsx', 'Info', index_col=None)
data_xls.to_csv('Info.csv', encoding='utf-8',index = False)
data_xls = pd.read_excel('Training data.xlsx', 'TPR', index_col=None)
data_xls.to_csv('TPR.csv', encoding='utf-8', index = False)

infoSet = pd.read_csv("Info.csv", header = None, skiprows=[0])
TPRSet = pd.read_csv("TPR.csv", header = None, skiprows=[0])

TPRSet


# In[79]:


# Merge info set and TPR set
from sklearn.decomposition import PCA

aggregation_functions = {2: 'mean', 3: 'mean', 4: 'mean', 5: 'mean', 6: 'mean'}

TPRaggreg = TPRSet.groupby(TPRSet[0]).aggregate(aggregation_functions)
mergeSet = pd.merge(infoSet.iloc[:, [True, True, True, False, False, False, False]], TPRaggreg.iloc[:, [True, True, True, True, True]], on=0)
mergeSet


# In[80]:


# Extract feature list and clean strings
import numpy as np

def str_trim(series):
    series = series.replace(' ', np.nan)
    series = series.replace('', np.nan, regex=True)
    series = series.replace("0", np.nan, regex=True)
    
    feature = pd.DataFrame(series.str.split(',').explode())
    feature = feature.drop_duplicates(subset=None, keep='first', inplace=False)
    feature = feature.replace('', np.nan)
    feature = feature.dropna()

    return series, feature

bacteria = infoSet.iloc[:, 5]
bacteria = bacteria.rename("bacteria")
bacteriaSet, bacteria_feature= str_trim(bacteria)
bacteria_feature = bacteria_feature.to_numpy().flatten()

common = infoSet.iloc[:, 3]
common = common.rename("common")
commonSet, commonFeature = str_trim(common)
commonFeature = commonFeature.to_numpy().flatten()
# mergeSet[5]


# In[81]:


# Create one hot encoding
def extract_df(ser, features):
    df= pd.DataFrame()
    for feature in features:
        id = 0
        new_list = []
        for element in ser:
            if feature == element:
                new_list.append(1)
            else:
                new_list.append(0)
    
        df[feature] = new_list
    return df
            
#origin_ant
bacteria_df = extract_df(bacteriaSet, bacteria_feature)
common_df = extract_df(commonSet, commonFeature)
common_df


# In[82]:


# Create cleaned data set
import numpy as np
from sklearn import preprocessing


# 創造 dummy variables
label_encoder = preprocessing.LabelEncoder()
encoded_no = label_encoder.fit_transform(mergeSet[0])

# 建立訓練與測試資料
patient_x = pd.concat([mergeSet, bacteria_df, common_df], axis=1)
patient_x = patient_x.iloc[:, 1:]
patient_y = infoSet.iloc[:, 6]

patient_x
# type(patient_x)


# In[83]:


# Reduce demensionality
from sklearn.feature_selection import SelectKBest, f_classif, chi2, SelectFpr, mutual_info_classif


kb = SelectKBest(mutual_info_classif, k=10).fit(patient_x, patient_y)
patient_x = kb.transform(patient_x)
patient_x


# In[84]:


# Seperate training set and testing set

from sklearn import model_selection
train_x, test_x, train_y, test_y = model_selection.train_test_split(patient_x, patient_y, test_size = 0.2)


# In[85]:


from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score

# Create models
forest = RandomForestClassifier(n_estimators = 100)
extraTrees = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
svc = svm.SVC(probability=True)
voting = VotingClassifier(estimators=[('rf', forest), ('et', extraTrees), ('svm', svc)], voting='soft')

for clf, label in zip([forest, extraTrees, svc, voting], ['Random Forest', 'Extra Trees', 'SVM', 'Voting Classifier']):
    scores = cross_val_score(clf, patient_x, patient_y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    
voting_fit = voting.fit(train_x, train_y)

# Predict
test_y_predicted = voting.predict(test_x)


# In[86]:


# Cal AUC
fpr, tpr, thresholds = metrics.roc_curve(test_y, test_y_predicted)
auc = metrics.auc(fpr, tpr)
print(auc)


# In[87]:


# f1 score
f1 = metrics.f1_score(test_y, test_y_predicted)
f1


# In[89]:


# Testing
data_xls = pd.read_excel('Testing data.xlsx', 'Info', index_col=None)
data_xls.to_csv('TestingInfo.csv', encoding='utf-8',index = False)
data_xls = pd.read_excel('Testing data.xlsx', 'TPR', index_col=None)
data_xls.to_csv('TestingTPR.csv', encoding='utf-8', index = False)

infoSet = pd.read_csv("TestingInfo.csv", header = None, skiprows=[0])
TPRSet = pd.read_csv("TestingTPR.csv", header = None, skiprows=[0])

aggregation_functions = {2: 'mean', 3: 'mean', 4: 'mean', 5: 'mean', 6: 'mean'}

TPRaggreg = TPRSet.groupby(TPRSet[0]).aggregate(aggregation_functions)
mergeSet = pd.merge(infoSet.iloc[:, [True, True, True, False, False, False]], TPRaggreg.iloc[:, [True, True, True, True, True]], on=0)

# Extract feature
bacteria = infoSet.iloc[:, 5]
bacteria = bacteria.rename("bacteria")
bacteriaSet, bacteria_feature_not_used = str_trim(bacteria)
# bacteria_feature_not_used = bacteria_feature_not_used.to_numpy().flatten()

common = infoSet.iloc[:, 3]
common = common.rename("common")
commonSet, commonFeature = str_trim(common)
commonFeature = commonFeature.to_numpy().flatten()

# One Hot encoding
bacteria_df = extract_df(bacteriaSet, bacteria_feature)
common_df = extract_df(commonSet, commonFeature)

# Concate
test_x = pd.concat([mergeSet, bacteria_df, common_df], axis=1)
test_x = test_x.iloc[:, 1:]

# Cover select k best mask
mask = kb.get_support()

test_x = test_x.iloc[:,mask]

test_y_predicted = voting.predict(test_x)

test_y_predicted


# In[ ]:





# In[ ]:




