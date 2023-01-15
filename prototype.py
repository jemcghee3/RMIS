import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Define a function to change the value of decision column
def change_decision_value(value):
    if value == 'accept':
        return 1
    else:
        return 0
    
def change_plusminus_value(value):
    if value == '++':
        return 2
    elif value == '+':
        return 1
    elif value == '-':
        return -1
    else:
        return 0

# this section loads the data and clears rows where the decision is not made

# Load the data from an Excel file
data = pd.read_excel('Registrations_Sep_2022_a.xlsm', header=2) # header = 2 means that the first two rows are not included in the data
# drop the columns with all NaN values
data = data.dropna(axis=1, how='all')
# drop the rows with all NaN values
data = data.dropna(axis=0, how='all')
# drop the rows with NaN values in the decision column
data = data.dropna(subset=['HoP'])
data = data.dropna(subset=['Deputy HoP'])
data.columns = [x.replace("\n", " ") for x in data.columns.to_list()]

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

# change the values of decision columns to 1 and 0
data['HoP'] = data['HoP'].apply(change_decision_value)
data['Deputy HoP'] = data['Deputy HoP'].apply(change_decision_value)
# merge the decision columns into one
data['Decision'] = data['HoP'] + data['Deputy HoP']
# drop the decision columns
data = data.drop(columns=['HoP', 'Deputy HoP'])
# change the decision column to 1 if the value is 2 and 0 otherwise
data['Decision'] = data['Decision'].apply(lambda x: 1 if x == 2 else 0)
# this means we want those applicants where both HoP and Deputy HoP have accepted


# change the values of plusminus columns to 2, 1, 0, -1
data['Grade (++, +, -, 0)'] = data['Grade (++, +, -, 0)'].apply(change_plusminus_value)
data['Experience (++, +, -, 0)'] = data['Experience (++, +, -, 0)'].apply(change_plusminus_value)



print('-----------------')



# Define the target variable and the parameters)


X = data.drop(columns=['Decision'])
y = data['Decision']

print(X)
print(y)
print('-----------------')

    

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Perform logistic regression
clf = LogisticRegression()
clf.fit(X_train[['Grade (++, +, -, 0)', 'Experience (++, +, -, 0)']], y_train)

# Print the accuracy of the model
print(clf.score(X_test[['Grade (++, +, -, 0)', 'Experience (++, +, -, 0)']], y_test))
# Print the class probabilities
print(clf.predict_proba(X_test[['Grade (++, +, -, 0)', 'Experience (++, +, -, 0)']]))
# Print the predicted classes
print(clf.predict(X_test[['Grade (++, +, -, 0)', 'Experience (++, +, -, 0)']]))
# Print the confusion matrix
print(pd.crosstab(y_test, clf.predict(X_test[['Grade (++, +, -, 0)', 'Experience (++, +, -, 0)']])))
# Print the X_test values where the predicted class is 1
print(X_test[clf.predict(X_test[['Grade (++, +, -, 0)', 'Experience (++, +, -, 0)']]) == 1])
# Print the X_test values where the predicted class is 0
print(X_test[clf.predict(X_test[['Grade (++, +, -, 0)', 'Experience (++, +, -, 0)']]) == 0])

"""
# Create a list to store the column 4 of training set
column_4_list = []

# Iterate through the training set
for i in range(len(X_train)):
    # Check if column 8 of the training set is 1
    if y_train.iloc[i] == 1:
        # Check if column 4 of the training set is in the list
        if X_train.iloc[i, 4] not in column_4_list:
            column_4_list.append(X_train.iloc[i, 4])
            
# Iterate through the test set
for i in range(len(X_test)):
    # Check if column 4 of the test set is in the list
    if X_test.iloc[i, 4] not in column_4_list:
        # Print the content of column 4 for that row
        print(X_test.iloc[i, 4])
        # Ask the user for input to accept or reject
        user_input = input("Accept or reject this value? (A/R)")
        if user_input == 'A':
            column_4_list.append(X_test.iloc[i, 4])
            
            
            
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Define the hyperparameters to search over
param_grid = {'C': [0.1, 1, 10],
              'penalty': ['l1', 'l2']}

# Create a logistic regression model
clf = LogisticRegression()

# Use GridSearchCV to perform a k-fold cross validation for each combination of hyperparameters
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# The best hyperparameters are stored in the best_params_ attribute
print(grid_search.best_params_)
"""