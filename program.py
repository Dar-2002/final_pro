import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

#reading dataset
df=pd.read_csv("diabetes.csv")

# crucial attributes
crucial_attributes=['gender','age','weight','num_medications','number_outpatient','number_emergency','number_inpatient','diag_1','diag_2','diag_3','A1Cresult','metformin','repaglinide','diabetesMed']
ds=df[crucial_attributes]


# gender encoding
lb=LabelEncoder()
ds['gender']=lb.fit_transform(ds['gender'])
ds['diabetesMed']=lb.fit_transform(ds['diabetesMed'])




#age mapping
age_map = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95}
ds['age'] = ds['age'].map(age_map)

#weight mapping
weight_map = {'[0-25)': 12.5, '[100-125)': 162.5, '[125-150)': 137.5, '[150-175)':162.5}
ds['weight'] = ds['weight'].map(weight_map)

#weight mean
mean_weight = ds['weight'].mean()
ds['weight'].fillna(mean_weight, inplace=True)



ds.replace("?", pd.NA, inplace=True)






#handle missing values of weight attribute by replacing them with mean
'''mean_weight = ds['weight'].mean()
ds['weight'].fillna(mean_weight, inplace=True)'''
# Define a function to map weight values based on age ranges
'''def map_weight_by_age(age):
    if age <= 10:
        return 25  # Example mapped weight for ages 0-10
    elif age <= 20:
        return 40  # Example mapped weight for ages 11-20
    elif age<=40:
        return 60
    else:
        return 70# Default value for other ages
ds['weight']=ds['age'].apply(map_weight_by_age)'''










print(ds)