#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import matthews_corrcoef
import random
from imblearn.under_sampling import RandomUnderSampler


# In[3]:


def run_pipeline(df, organism_specific_data, model_choice="Random Forest"):
    # Generate a binary vector for filtering
    selected_ids = [random.randint(0, 1) for _ in range(len(df))]

    # Separate rows with 0s and 1s
    selected_zeros = [i for i, value in enumerate(selected_ids) if value == 0]
    selected_ones = [i for i, value in enumerate(selected_ids) if value == 1]

    # Filter the dataset based on the selected_ids binary vector
    df_filtered = df.iloc[selected_ones]

    # Split organism_specific_data data into training and validation sets
    groups = organism_specific_data['Info_cluster']
    features = organism_specific_data.drop('Class', axis=1)
    labels = organism_specific_data['Class']

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_indices, validation_indices = next(splitter.split(features, labels, groups))

    organism_specific_data_train = organism_specific_data.iloc[train_indices]
    organism_specific_data_validation = organism_specific_data.iloc[validation_indices]

    # Concatenate organism_specific_data train data with the filtered data
    final_train = pd.concat([df_filtered, organism_specific_data_train], ignore_index=True)

    # Perform class rebalancing by undersampling the majority class
    under_sampler = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = under_sampler.fit_resample(final_train.drop("Class", axis=1), final_train["Class"])
    final_train_balanced = pd.DataFrame(X_train_resampled, columns=final_train.columns[:-1])
    final_train_balanced["Class"] = y_train_resampled

    # Check for duplicates and remove them
    final_train_balanced = pd.concat([final_train_balanced, organism_specific_data_train]).drop_duplicates().reset_index(drop=True)

    # Shuffle the final balanced training data
    final_train_balanced = final_train_balanced.sample(frac=1, random_state=42)

    # Define the target variables for training and validation
    y_train = final_train_balanced["Class"]
    y_val = organism_specific_data_validation["Class"]

    # Remove the "Class" column from the datasets
    X_train = final_train_balanced.drop(columns="Class")
    X_val = organism_specific_data_validation.drop(columns="Class")

    # Train and evaluate the selected model
    if model_choice == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(random_state=42)
    elif model_choice == "Support Vector Machine":
        model = SVC(random_state=42)
    else:
        print("Invalid model choice. Running Random Forest by default.")
        model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    # Calculate the MCC value for the selected model
    mcc = matthews_corrcoef(y_val, predictions)

    # Return the MCC value
    return mcc


# In[ ]:




