import pandas as pd
import numpy as np
from sklearn.utils import resample



df = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

def preprocess(dataset, nan_values, cat_to_numerical=True, upsample=False):
    """
    

    Parameters
    ----------
    dataset : TYPE
        The dataset to preprocess.
    nan_values : TYPE
        How to deal with nan values. The possible options are:
            - "drop columns": all columns that contains nan values are removed
            - "drop rows": all rows that contains nan values are removed
            - "replace with 0": nan values are replaced with a 0
            - "replace with median": nan values are replaced with the median of the whole column.
    cat_to_numerical : TYPE, optional
        Convert all categorical values in numerical. The default is True.
    upsample : TYPE, optional
        Upsample the dataset by randomly duplicating instances whose target variable belong to the minority 
        class ("smoke" -> 1) in order to avoid an unbalanced dataset. The default is False.

    Returns
    -------
    dataset : TYPE
        The preprocessed dataset.

    """
    
    
    if upsample==True:
        stroke_data = dataset[dataset["stroke"] ==1]
        no_stroke_data = dataset[dataset["stroke"] == 0]
        no_stroke_upsampled = resample(stroke_data, replace=True, n_samples=len(no_stroke_data), random_state=133)
        balanced_data = pd.concat([no_stroke_data, no_stroke_upsampled])
        dataset = balanced_data.sample(frac=1, random_state=133)


    
    if cat_to_numerical==True:
        # convert categorical columns into numerical
        categorical = ["gender", "work_type", "Residence_type", "smoking_status"]
        
        for col in categorical:
            unique_values = dataset[col].unique()
            # put smoking status in order
            smoking_cats = ['Unknown', 'never smoked', 'formerly smoked','smokes']
            if col == "smoking_status":
                dataset[col] = dataset[col].replace(smoking_cats, [0,1,2,3])           
            else:
                dataset[col] = dataset[col].replace(unique_values, list(range(len(unique_values))))
    
    if nan_values == "drop columns":
        dataset = dataset.dropna(1)
    elif nan_values == "drop rows":
        dataset = dataset.dropna(0)
    elif nan_values == "replace with 0s":
        dataset = dataset.replace(np.nan, 0)
    elif nan_values == "replace with median":
        # The standard deviation in the data is high, so it is decided to use the Median.
        # The mean is not used to prevent outliers from over-altering the data.
        bmi_median = round(df["bmi"].median(), 2)
        glucose_median = round(df["avg_glucose_level"].median(), 4)
        dataset["bmi"] = dataset["bmi"].fillna(bmi_median)
        dataset["avg_glucose_level"] = dataset["avg_glucose_level"].fillna(glucose_median)
    return dataset

df_preprocessed = preprocess(df, "replace with median", upsample = True)