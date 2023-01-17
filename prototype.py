import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# Define a function to perform five-fold cross validation
def five_fold_cv(model, X, y):
    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        # print(model.score(X_test, y_test))
        results.append(model.score(X_test, y_test))
    print("The average cross-validation accuracy is ", sum(results) / len(results))
        
# Define a function to change the value of decision column
def change_decision_value(value):
    if value == 'accept':
        return 1
    else:
        return 0

# Define a function to change the value of plusminus columns
def change_plusminus_value(value):
    if value == '++':
        return 2
    elif value == '+':
        return 1
    elif value == '-':
        return -1
    else:
        return 0

# add rules of what happens when -1 and when average is less than 1.

# this section loads the data and clears rows where the decision is not made

# Load the data from an Excel file
data = pd.read_excel('./Registrations_Sep_2022_a.xlsm', header=2) # header = 2 means that the first two rows are not included in the data
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



# Define the target variable and the parameters)


X = data.drop(columns=['Decision'])
y = data['Decision']


clf = LogisticRegression(random_state=1)
five_fold_cv(clf, X[['Grade (++, +, -, 0)', 'Experience (++, +, -, 0)']], y)


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Perform logistic regression
clf = LogisticRegression(random_state=1)
clf.fit(X_train[['Grade (++, +, -, 0)', 'Experience (++, +, -, 0)']], y_train)

# Print the accuracy of the model
print(clf.score(X_test[['Grade (++, +, -, 0)', 'Experience (++, +, -, 0)']], y_test))
# Print the class probabilities
print(clf.predict_proba(X_test[['Grade (++, +, -, 0)', 'Experience (++, +, -, 0)']])) # the second item in the sublist is the probability of the class 1
# save the predicted classes
predicted_classes = clf.predict(X_test[['Grade (++, +, -, 0)', 'Experience (++, +, -, 0)']])
X_test['Predicted'] = predicted_classes
# Print the predicted classes
# print(predicted_classes)
# Print the confusion matrix
print(pd.crosstab(y_test, predicted_classes, rownames=['Actual'], colnames=['Predicted'], margins=True))
# Print the X_test values where the predicted class is 1
print("After applying the machine learning model, the predicted class for the following students is 1, that the student would be accepted:")
print(X_test[predicted_classes == 1])
# Print the X_test values where the predicted class is 0
print("After applying the machine learning model, the predicted class for the following students is 1, that the student would be accepted:")
print(X_test[predicted_classes == 0])


# Create a list to store the approved universities of training set
approved_universities = []

# Iterate through the training set to find the approved universities in the column named "University" in row i
X_train.columns.get_loc("University")
# Iterate through the training set
for i in range(len(X_train)):
    # Check if acceptance of the training set is 1
    if y_train.iloc[i] == 1:
        # Check if university of the training set is in the list
        if X_train.iloc[i, X_train.columns.get_loc("University")] not in approved_universities:
            approved_universities.append(X_train.iloc[i, X_train.columns.get_loc("University")])

# Iterate through the test set
print("The following universities are not in the list of previously-approved universities. Please check whether they are valid or not.")
for i in range(len(X_test)):
    # Check if university of the test set is in the list
    if X_test.iloc[i, X_test.columns.get_loc("University")] not in approved_universities:
        # Print the content of university column for that row
        while True:
            print(X_test.iloc[i, X_test.columns.get_loc("University")])
            # Ask the user for input to accept or reject
            user_input = input("Accept or reject this value? (A/R) ")
            try : 
                user_input = user_input.upper()
                # Check if the input is valid
                if user_input != 'A' and user_input != 'R':
                    raise ValueError
                break
            except ValueError:
                print("Invalid input")
        if user_input == 'A':
            approved_universities.append(X_test.iloc[i, X_test.columns.get_loc("University")])
        else:
            predicted_classes[i] = 0 # Reject the student

print("After applying a rule on whether the university is approved and checking with the human, the following students should be rejected:")
for i in range(len(predicted_classes)):
 
# update the decision column
    if X_test.iloc[i, X_test.columns.get_loc("Predicted")] == 1 and predicted_classes[i] == 0:
        print("Rejecting the student due to the university not being approved.")
        print(X_test.iloc[[i]])
        X_test.iloc[i, X_test.columns.get_loc("Predicted")] = predicted_classes[i]
#for i in range(len(predicted_classes)):
#    X_test.iloc[i, X_test.columns.get_loc("Decision")] = predicted_classes[i]

print("After applying the rule on whether the university is approved and checking with the human, the following students would be rejected:")
print(X_test[predicted_classes == 0])


            
# Need to reject the student if the university is not in the list - done
# save approved universities to a file
# if a student has a -1, then the student is rejected
# if the grade + experience is less than 1, then the student is rejected