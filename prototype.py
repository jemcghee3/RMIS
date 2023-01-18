import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import pickle
import fhnw_KB as KB

# Define a function to perform five-fold cross validation
def five_fold_cv(model, X, y):
    # this function takes a model, a dataframe of features, and a series of labels
    # it performs five-fold cross validation and returns the average accuracy
    # the model is trained on four-fifths of the data and tested on the remaining one-fifth
    # the process is repeated five times
    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        # print(model.score(X_test, y_test))
        results.append(model.score(X_test, y_test))
    print("The average cross-validation accuracy is ", sum(results) / len(results))
        





# Load the data from an Excel file
# 2023_spring and 2022_fall to be used for test data
# df_2023_spring = pd.read_excel('./Registrations_Feb_2023_a.xlsx', header=2) # header = 2 means that the first two rows are not included in the data
# df_2022_fall = pd.read_excel('./Registrations_Sep_2022_a.xlsm', header=2) # header = 2 means that the first two rows are not included in the data
df_2022_spring = pd.read_excel('./Registrations_Feb_2022_a.xlsm', header=2) # header = 2 means that the first two rows are not included in the data
df_2021_fall = pd.read_excel('./Registrations_Sep_2021_a.xlsm', header=2) # header = 2 means that the first two rows are not included in the data
df_2021_spring = pd.read_excel('./Registrations_Feb_2021_a.xlsm', header=2) # header = 2 means that the first two rows are not included in the data
df_2020_fall = pd.read_excel('./Registrations_Sep_2020_a.xlsm', header=2) # header = 2 means that the first two rows are not included in the data

data = pd.concat([df_2022_spring, df_2021_fall, df_2021_spring, df_2020_fall], axis = 0)

data = KB.clean_data(data)


"""
for col in data.columns:
    print(col)
"""

"""
Unnamed: 0
Last Name
First Name
Start
FT
PT
Gender
Bachelor
University
University Country
University (H+, H-, H+/-)
Official Grade
Grade
Grade (++, +, -, 0) # column 13
Experience (y,m)
Experience (++, +, -, 0) # column 15
English
Interview
Status
Condition
BPM
BA
IS
Citizenship
Canton of residence
Visum?
Remarks
Spalte1
Spalte12
Spalte13
Spalte14
Spalte15
Decision
"""




# Define the target variable and the parameters)


X = data.drop(columns=['Decision'])
y = data['Decision']


clf = LogisticRegression(random_state=1)
five_fold_cv(clf, X[['Grade (++, +, -, 0)', 'Experience (++, +, -, 0)']], y)



# Perform logistic regression
clf = LogisticRegression(random_state=1)
clf.fit(X[['Grade (++, +, -, 0)', 'Experience (++, +, -, 0)']], y)

# save the model to disk
pickle.dump(clf, open('model.pkl','wb'))

# Create a list to store the approved universities of training set
approved_universities = []

# Iterate through the training set to find the approved universities in the column named "University" in row i
X.columns.get_loc("University")
# Iterate through the training set
for i in range(len(X)):
    # Check if acceptance of the training set is 1
    if y.iloc[i] == 1:
        # Check if university of the training set is in the list
        if X.iloc[i, X.columns.get_loc("University")] not in approved_universities:
            approved_universities.append(X.iloc[i, X.columns.get_loc("University")])

# Create a file to store the approved universities of training set
pickle.dump(approved_universities, open('approved_universities.pkl', 'wb'))
print('Done writing approved universities into a binary file')