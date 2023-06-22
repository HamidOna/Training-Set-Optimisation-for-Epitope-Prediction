#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef


# In[2]:


def run_pipeline(df, lentivirus_data, class_weights):
    # Drop rows with missing values
    df.dropna(inplace=True)

    # Drop columns starting with "Info_" except "Info_cluster"
    columns_to_drop = df.filter(regex='^Info_(?!cluster)').columns
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    selected_features = ['Info_cluster','feat_esm1b_15','feat_esm1b_38','feat_esm1b_46','feat_esm1b_69','feat_esm1b_70',
         'feat_esm1b_76',
         'feat_esm1b_87',
         'feat_esm1b_89',
         'feat_esm1b_125',
         'feat_esm1b_128',
         'feat_esm1b_141',
         'feat_esm1b_162',
         'feat_esm1b_179',
         'feat_esm1b_188',
         'feat_esm1b_197',
         'feat_esm1b_213',
         'feat_esm1b_217',
         'feat_esm1b_224',
         'feat_esm1b_252',
         'feat_esm1b_254',
         'feat_esm1b_255',
         'feat_esm1b_265',
         'feat_esm1b_267',
         'feat_esm1b_279',
         'feat_esm1b_290',
         'feat_esm1b_302',
         'feat_esm1b_303',
         'feat_esm1b_304',
         'feat_esm1b_308',
         'feat_esm1b_313',
         'feat_esm1b_324',
         'feat_esm1b_330',
         'feat_esm1b_331',
         'feat_esm1b_343',
         'feat_esm1b_344',
         'feat_esm1b_358',
         'feat_esm1b_367',
         'feat_esm1b_392',
         'feat_esm1b_397',
         'feat_esm1b_413',
         'feat_esm1b_423',
         'feat_esm1b_434',
         'feat_esm1b_447',
         'feat_esm1b_450',
         'feat_esm1b_455',
         'feat_esm1b_457',
         'feat_esm1b_459',
         'feat_esm1b_472',
         'feat_esm1b_474',
         'feat_esm1b_476',
         'feat_esm1b_484',
         'feat_esm1b_487',
         'feat_esm1b_494',
         'feat_esm1b_500',
         'feat_esm1b_509',
         'feat_esm1b_526',
         'feat_esm1b_535',
         'feat_esm1b_541',
         'feat_esm1b_564',
         'feat_esm1b_570',
         'feat_esm1b_600',
         'feat_esm1b_621',
         'feat_esm1b_628',
         'feat_esm1b_639',
         'feat_esm1b_643',
         'feat_esm1b_646',
         'feat_esm1b_659',
         'feat_esm1b_665',
         'feat_esm1b_668',
         'feat_esm1b_669',
         'feat_esm1b_670',
         'feat_esm1b_671',
         'feat_esm1b_679',
         'feat_esm1b_684',
         'feat_esm1b_699',
         'feat_esm1b_711',
         'feat_esm1b_725',
         'feat_esm1b_728',
         'feat_esm1b_733',
         'feat_esm1b_741',
         'feat_esm1b_757',
         'feat_esm1b_771',
         'feat_esm1b_777',
         'feat_esm1b_785',
         'feat_esm1b_789',
         'feat_esm1b_795',
         'feat_esm1b_801',
         'feat_esm1b_805',
         'feat_esm1b_810',
         'feat_esm1b_847',
         'feat_esm1b_854',
         'feat_esm1b_874',
         'feat_esm1b_877',
         'feat_esm1b_882',
         'feat_esm1b_884',
         'feat_esm1b_898',
         'feat_esm1b_904',
         'feat_esm1b_909',
         'feat_esm1b_927',
         'feat_esm1b_928',
         'feat_esm1b_929',
         'feat_esm1b_933',
         'feat_esm1b_936',
         'feat_esm1b_942',
         'feat_esm1b_960',
         'feat_esm1b_966',
         'feat_esm1b_1015',
         'feat_esm1b_1043',
         'feat_esm1b_1050',
         'feat_esm1b_1056',
         'feat_esm1b_1103',
         'feat_esm1b_1109',
         'feat_esm1b_1110',
         'feat_esm1b_1125',
         'feat_esm1b_1132',
         'feat_esm1b_1148',
         'feat_esm1b_1153',
         'feat_esm1b_1156',
         'feat_esm1b_1159',
         'feat_esm1b_1166',
         'feat_esm1b_1167',
         'feat_esm1b_1180',
         'feat_esm1b_1183',
         'feat_esm1b_1186',
         'feat_esm1b_1208',
         'feat_esm1b_1216',
         'feat_esm1b_1223',
         'feat_esm1b_1231',
         'feat_esm1b_1233',
         'feat_esm1b_1237',
         'feat_esm1b_1239',
         'feat_esm1b_1240',
         'feat_esm1b_1274',
         'Class']

    df = df[selected_features]
    # Standardize the dataset
    scaler = StandardScaler()
    mask = df.columns != "Class"
    df.loc[:, mask] = scaler.fit_transform(df.loc[:, mask])

    lentivirus = lentivirus_data[selected_features]
    mask = lentivirus.columns != "Class"
    lentivirus.loc[:, mask] = scaler.transform(lentivirus.loc[:, mask])

 
    lentivirus_train, lentivirus_test = train_test_split(lentivirus, test_size=0.25, random_state=42)

    # Concatenate lentivirus train data with the training data
    final_train = pd.concat([df, lentivirus_train], ignore_index=True)

    # Define the target variables for training and testing
    y_train = final_train["Class"]
    y_test = lentivirus_test["Class"]

    # Remove the "Class" column from the datasets
    X_train = final_train.drop(columns="Class")
    X_test = lentivirus_test.drop(columns="Class")

    # Train the models
    rf_model = RandomForestClassifier(class_weight=class_weights, random_state=42)
    lr_model = LogisticRegression(class_weight=class_weights, random_state=42)
    svm_model = SVC(class_weight=class_weights, random_state=42)

    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)

    # Make predictions on the testing set
    rf_predictions = rf_model.predict(X_test)
    lr_predictions = lr_model.predict(X_test)
    svm_predictions = svm_model.predict(X_test)

    # Calculate the MCC value for each model
    rf_mcc = matthews_corrcoef(y_test, rf_predictions)
    lr_mcc = matthews_corrcoef(y_test, lr_predictions)
    svm_mcc = matthews_corrcoef(y_test, svm_predictions)

    # Return the MCC values
    return rf_mcc, lr_mcc, svm_mcc



# In[ ]:




