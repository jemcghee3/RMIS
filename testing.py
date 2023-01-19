import pandas as pd
import numpy as np
import pickle
import fhnw_KB as KB
from sklearn.metrics import roc_auc_score

# Load the data from an Excel file
# 2023_spring and 2022_fall to be used for test data
df_2023_spring = pd.read_excel('./Registrations_Feb_2023_a.xlsx', header=2) # header = 2 means that the first two rows are not included in the data
df_2022_fall = pd.read_excel('./Registrations_Sep_2022_a.xlsm', header=2) # header = 2 means that the first two rows are not included in the data
test_data = pd.concat([df_2023_spring, df_2022_fall], axis = 0)

test_data = KB.clean_data(test_data)

X_test = test_data.drop(columns=['Decision'])
y_test = test_data['Decision']


clf = pickle.load(open('model.pkl','rb'))


# Print the accuracy of the model
# print(clf.score(X_test[['Grade (++, +, -, 0)', 'Experience (++, +, -, 0)']], y_test))
# Print the class probabilities
# print(clf.predict_proba(X_test[['Grade (++, +, -, 0)', 'Experience (++, +, -, 0)']])) # the second item in the sublist is the probability of the class 1
# save the predicted classes
ML_predictions = clf.predict(X_test[['Grade (++, +, -, 0)', 'Experience (++, +, -, 0)']])
# X_test['Predicted'] = ML_predictions
# Print the predicted classes
# print(ML_predictions)
# Print the confusion matrix
# print(pd.crosstab(y_test, ML_predictions, rownames=['Actual'], colnames=['Predicted'], margins=True))
# Print the X_test values where the predicted class is 1
# print("After applying the machine learning model, the predicted class for the following students is 1, that the student would be accepted:")
# print(X_test[ML_predictions == 1])
# Print the X_test values where the predicted class is 0
# print("After applying the machine learning model, the predicted class for the following students is 0, that the student would be rejected:")
# print(X_test[ML_predictions == 0])




# The code below is for applying the model to new applicants
loop_decisions = KB.loop_decision_maker(X_test)
KB_decisions = KB.KB_decision_maker(X_test)

approved_universities = pickle.load(open('approved_universities.pkl', 'rb'))

# Iterate through the test set
print("The following universities are not in the list of previously-approved universities. Please check whether they are valid or not.")
# commented out for testing purposes to avoid having to answer each time
for i in range(len(X_test)):
    result, approved_universities = KB.check_university(X_test.iloc[i, X_test.columns.get_loc("University")], approved_universities)
    if result == 0:
        print("The student {} {} is rejected. {} is not an approved university.".format(X_test.iloc[i, X_test.columns.get_loc("First Name")], X_test.iloc[i, X_test.columns.get_loc("Last Name")], X_test.iloc[i, X_test.columns.get_loc("University")]))
        KB_decisions[i] = 0

compensation_rule_check = False
for i in range(len(X_test)):
    result = KB.compare_grade_experience(X_test.iloc[i, X_test.columns.get_loc("Grade (++, +, -, 0)")], X_test.iloc[i, X_test.columns.get_loc("Experience (++, +, -, 0)")])
    if result == 0: # Only check the rule if the student would be rejected by the rule
        if compensation_rule_check == False:
            rule = "One '0' for either average Bachelor's grade or for experience can be compensated by one '++' in another criterion."
            compensation_rule_check, compensation_rule_validity = KB.rule_checker(rule)
            if compensation_rule_validity == False:
                break # If the rule is not valid, then the loop is broken to avoid applications being rejected under the rule
        print("The student {} {} is rejected. The grade score was {} and the experience score was {}.".format(X_test.iloc[i, X_test.columns.get_loc("First Name")], X_test.iloc[i, X_test.columns.get_loc("Last Name")], X_test.iloc[i, X_test.columns.get_loc("Grade (++, +, -, 0)")], X_test.iloc[i, X_test.columns.get_loc("Experience (++, +, -, 0)")]))
        KB_decisions[i] = 0

no_negative_score_rule_check = False
for i in range(len(X_test)):
    result = KB.no_negative_score(X_test.iloc[i, X_test.columns.get_loc("Grade (++, +, -, 0)")])
    if result == 0:
        if no_negative_score_rule_check == False:
            rule = "A rating in any criterion with '-' constitutes a 'not acceptable'."
            no_negative_score_rule_check, no_negative_rule_validity = KB.rule_checker(rule)
            if no_negative_rule_validity == False:
                break # If the rule is not valid, then the loop is broken to avoid applications being rejected under the rule
        print("The student {} {} is rejected. The grade score was '-', which requires rejection.".format(X_test.iloc[i, X_test.columns.get_loc("First Name")], X_test.iloc[i, X_test.columns.get_loc("Last Name")]))
        KB_decisions[i] = 0
    result = KB.no_negative_score(X_test.iloc[i, X_test.columns.get_loc("Experience (++, +, -, 0)")])        
    if result == 0:
        if no_negative_score_rule_check == False:
            rule = "A rating in any criterion with '-' constitutes a 'not acceptable'."
            no_negative_score_rule_check, no_negative_rule_validity = KB.rule_checker(rule)
            if no_negative_rule_validity == False:
                break # If the rule is not valid, then the loop is broken to avoid applications being rejected under the rule
        print("The student {} {} is rejected. The experience score was '-', which requires rejection.".format(X_test.iloc[i, X_test.columns.get_loc("First Name")], X_test.iloc[i, X_test.columns.get_loc("Last Name")]))
        KB_decisions[i] = 0
        
# for i in range(len(X_test)):
    

for i in range(len(X_test)):
    loop_decisions[i] = min(ML_predictions[i], KB_decisions[i])

# print(ML_predictions)
# print(type(ML_predictions))

loop_decisions = np.array(loop_decisions)  

# print(type(loop_decisions))


print("After applying the entire loop, the predicted class for the following students is 1, that the student would be accepted:")
print(X_test[loop_decisions == 1])
# Print the X_test values where the predicted class is 0
print("After applying the entire loop, the predicted class for the following students is 0, that the student would be rejected:")
print(X_test[loop_decisions == 0])

# Print the final confusion matrix
print("The final confusion matrix for the entire loop is:")
print(pd.crosstab(y_test, loop_decisions, rownames=['Actual'], colnames=['Predicted'], margins=True))

print("The accuracy of the machine learning model is", clf.score(X_test[['Grade (++, +, -, 0)', 'Experience (++, +, -, 0)']], y_test))
print("The area under the Receiver Operating Characteristic Curve for the machine learning model is", roc_auc_score(y_test, clf.predict_proba(X_test[['Grade (++, +, -, 0)', 'Experience (++, +, -, 0)']])[:,1]))
KB_decisions = pd.Series(KB_decisions)
print("The accuracy of the knowledge base is", (sum(1 for x,y in zip(KB_decisions,y_test) if x == y) / len(y_test)))
print("The area under the Receiver Operating Characteristic Curve for the knowledge base is", roc_auc_score(y_test, KB_decisions))
print("The accuracy of the entire loop is", (sum(1 for x,y in zip(loop_decisions,y_test) if x == y) / len(y_test)))
print("The area under the Receiver Operating Characteristic Curve for the entire loop is", roc_auc_score(y_test, loop_decisions))

# make the predictions into columns in X_test and save the data to an excel file
X_test['ML Predicted'] = ML_predictions
X_test['KB Predicted'] = KB_decisions
X_test['Loop Predicted'] = loop_decisions
X_test.to_excel('test_data.xlsx')
print("The admissions decisions have been saved to the file 'test_data.xlsx'.")