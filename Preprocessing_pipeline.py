#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

def preprocessing_pipeline():
    # BorutaPy requires a base estimator, typically RandomForestClassifier
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    feature_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=1)

    # Step 5: Apply StandardScaler but exclude certain columns
    # Identify columns to scale
    scale_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Use ColumnTransformer to apply transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('scale', scale_transformer, lambda x: ~x.columns.isin(["Info_organism_id", "Info_cluster", "Class"]))
        ], 
        remainder='passthrough'
    )

    # Incorporate BorutaPy feature selector
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                               ('feature_selector', feature_selector)])

    return pipeline

def fit_data(df):
    # Preprocess
    df = df.dropna()
    columns_to_drop = [col for col in df.columns if col.startswith("Info_") and col not in ["Info_organism_id", "Info_cluster"]]
    X = df.drop(columns=['Class'] + columns_to_drop)
    y = df['Class']

    # Fit the pipeline
    pipeline = preprocessing_pipeline()
    pipeline.fit(X, y)
    return pipeline

def transformation_pipeline(df, pipeline):
    # Drop rows with missing data
    df = df.dropna().reset_index(drop=True)

    # store the columns to keep
    organism_id_series = df['Info_organism_id'].copy()
    cluster_series = df['Info_cluster'].copy()
    class_series = df['Class'].copy()

    columns_to_drop = [col for col in df.columns if col.startswith("Info_") and col not in ["Info_organism_id", "Info_cluster"]]
    X = df.drop(columns=['Class'] + columns_to_drop)

    # Use the pipeline to transform the dataset
    X_transformed = pipeline.transform(X)

    # Construct the transformed dataframe
    transformed_df = pd.DataFrame(X_transformed, columns=X.columns[pipeline.named_steps['feature_selector'].support_])

    # Add back the non-transformed columns
    transformed_df['Info_organism_id'] = organism_id_series
    transformed_df['Info_cluster'] = cluster_series
    transformed_df['Class'] = class_series

    return transformed_df

