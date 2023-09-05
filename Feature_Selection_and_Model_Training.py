#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef
from imblearn.under_sampling import RandomUnderSampler

def run_pipeline(df_filtered, organism_specific_data, model_choice="Random Forest"):
    # Split organism_specific_data data into training and validation sets
    groups = organism_specific_data['Info_cluster']
    
    features = organism_specific_data.drop(['Class', 'Info_cluster', 'Info_organism_id'], axis=1)
    labels = organism_specific_data['Class']

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25)
    train_indices, validation_indices = next(splitter.split(features, labels, groups))

    organism_specific_data_train = organism_specific_data.iloc[train_indices]
    organism_specific_data_validation = organism_specific_data.iloc[validation_indices]

    # Concatenate organism_specific_data train data with the filtered data
    final_train = pd.concat([df_filtered, organism_specific_data_train], ignore_index=True)

    under_sampler = RandomUnderSampler()
    X_train_resampled, y_train_resampled = under_sampler.fit_resample(final_train.drop("Class", axis=1), final_train["Class"])

    final_train_balanced = pd.DataFrame(X_train_resampled, columns=final_train.columns[:-1])
    final_train_balanced["Class"] = y_train_resampled

    final_train_balanced = pd.concat([final_train_balanced, organism_specific_data_train]).drop_duplicates().reset_index(drop=True)
    final_train_balanced = final_train_balanced.sample(frac=1)

    train_clusters = set(final_train_balanced['Info_cluster'])
    validation_clusters = set(organism_specific_data_validation['Info_cluster'])
    common_clusters = train_clusters.intersection(validation_clusters)

    final_train_balanced = final_train_balanced[~final_train_balanced['Info_cluster'].isin(common_clusters)]

    y_train = final_train_balanced["Class"]
    y_val = organism_specific_data_validation["Class"]

    # Dropping the 'Info_cluster' and 'Info_organism_id' columns as suggested
    X_train = final_train_balanced.drop(columns=["Class", "Info_cluster", "Info_organism_id"])
    X_val = organism_specific_data_validation.drop(columns=["Class", "Info_cluster", "Info_organism_id"])

    if model_choice == "Random Forest":
        model = RandomForestClassifier()
    elif model_choice == "KNN":
        model = KNeighborsClassifier()
    elif model_choice == "Support Vector Machine":
        model = SVC()
    else:
        print("Invalid model choice. Running Random Forest by default.")
        model = RandomForestClassifier()

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    mcc = matthews_corrcoef(y_val, predictions)
    return mcc
