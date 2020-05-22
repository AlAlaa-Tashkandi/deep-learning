import numpy as np
import pandas as pd

admissions = pd.read_csv('student_data.csv')

# Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)
display(data.head())
# Standarize features
for field in ['gre', 'gpa']:
    print('Field values are {}'.format(field))
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std

display(data.head())
# Split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
print(sample)
data, test_data = data.loc[sample], data.drop(sample)
display(data.head())


# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']
