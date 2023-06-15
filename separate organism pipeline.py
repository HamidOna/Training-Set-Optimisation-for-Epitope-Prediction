#!/usr/bin/env python
# coding: utf-8

# In[2]:


def extract_ids_by_scientific_name(dataset, scientific_name):
 
    # Filter the dataset based on "Scientific name" equals the provided scientific_name
    filtered_dataset = dataset[dataset["ScientificName"] == scientific_name]
    
    # Extract the values from the "IDs" column into a list
    id_list = filtered_dataset["IDs"].tolist()
    
    # Return the extracted ID values
    return id_list


# In[4]:


def separate_list_by_comma(lst):
    # Create an empty list to store the separated values
    separated_values = []
    
    # Iterate over each element in the input list
    for item in lst:
        # Split the item based on a comma
        values = item.split(',')
        
        # Convert the split values to integers
        values = [int(val) for val in values]
        
        # Extend the separated values list with the split values
        separated_values.extend(values)
    
    # Return the separated values as a list of integers
    return separated_values


# In[5]:


def remove_rows_by_ids(dataset, column_name, ids):
    # Create a boolean mask to identify rows with matching IDs
    mask = dataset[column_name].isin(ids)
    
    # Create new dataset from rows with matching IDs
    new_dataset = dataset[mask]
    new_dataset = new_dataset.reset_index(drop=True)

    # Remove rows with matching IDs from the dataset
    dataset = dataset[~mask]
    
    # Reset the index of the dataset
    dataset = dataset.reset_index(drop=True)
    
    
    
    # Return the updated dataset
    return new_dataset, dataset


# In[6]:


def separate_organisms_pipeline(dataset1, dataset2, scientific_name):
    # Step 1: Extract IDs by scientific name from dataset1
    id_list = extract_ids_by_scientific_name(dataset1, scientific_name)

    # Step 2: Convert ID list to a list of integers
    ids = separate_list_by_comma(id_list)

    # Step 3: Remove rows by IDs from dataset2
    new_dataset, updated_dataset2 = remove_rows_by_ids(dataset2, "IDs", ids)

    return new_dataset, updated_dataset2


# In[ ]:




