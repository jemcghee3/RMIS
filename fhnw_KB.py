import pandas as pd

def KB_decision_maker(new_applications):
    # this function takes a dataframe of new applications and returns a list of predictions to be made by the KB
    # the value is 1 because the KB has not yet excluded any applications
    KB_prediction = []
    for i in range(len(new_applications)):
        KB_prediction.append(1)
    return KB_prediction

def loop_decision_maker(new_applications):
    # this function takes a dataframe of new applications and returns a list of predictions to be made by the loop
    # the value is None because the loop has not yet made a prediction  
    loop_prediction = []
    for i in range(len(new_applications)):
        loop_prediction.append(None)
    return loop_prediction

def input_cleaner(new_value):
    # this function takes a value and asks the user to accept or reject it
    # if the user accepts the value, the function returns 'A'
    # if the user rejects the value, the function returns 'R'
    while True:
        print("Please verify whether the following value is valid:")
        print("The value is ", new_value)
        user_input = input("Accept or reject this value? (A/R) ")
        try : 
            user_input = user_input.upper()
            # Check if the input is valid
            if user_input != 'A' and user_input != 'R':
                raise ValueError
            return user_input
        except ValueError:
            print("Invalid input")
        
def check_university(applicant_university, approved_universities):
    # this function takes the university of an applicant and a list of approved universities
    # if the university is approved, the function returns 1
    # if the university is not approved, the function asks the user to accept or reject the university
    # if the user accepts the university, the function adds the university to the list of approved universities and returns 1
    # if the user rejects the university, the function returns 0
    if applicant_university not in approved_universities:
        human_input = input_cleaner(applicant_university)
        if human_input == 'A':
            approved_universities.append(applicant_university)
            return 1, approved_universities # return 1 if the university is approved because this does not exclude the application
        else:
            return 0, approved_universities # return 0 if the university is rejected because this rejects the application
    return 1, approved_universities # if the university is already approved, return 1 because this does not exclude the application

# Define a function to change the value of decision column
# data cleaning
def change_decision_value(value):
    # this function takes a value in the decision column and returns 1 if the value is 'accept' and 0 otherwise
    if value == 'accept':
        return 1
    else:
        return 0

# Define a function to change the value of plusminus columns
# data cleaning
def change_plusminus_value(value):
    # this function takes a value in the plusminus columns and returns 2 if the value is '++', 1 if the value is '+', -1 if the value is '-', and 0 otherwise
    if value == '++':
        return 2
    elif value == '+':
        return 1
    elif value == '-':
        return -1
    else:
        return 0
    
def clean_data(df):
    # this function takes a dataframe and returns a cleaned dataframe
    # the function drops the columns that are not used in the model
    # the function changes the values in the decision column to 1 and 0
    # the function changes the values in the plusminus columns to 2, 1, -1, and 0
    # the function drops the rows where the decision is not made
    # drop the columns with all NaN values
    df = df.dropna(axis=1, how='all')
    # drop the rows with all NaN values
    df = df.dropna(axis=0, how='all')
    # drop the rows with NaN values in the decision column
    df = df.dropna(subset=['HoP'])
    df = df.dropna(subset=['Deputy HoP'])
    df.columns = [x.replace("\n", " ") for x in df.columns.to_list()]
    # change the values of decision columns to 1 and 0
    df['HoP'] = df['HoP'].apply(change_decision_value)
    df['Deputy HoP'] = df['Deputy HoP'].apply(change_decision_value)
    # merge the decision columns into one
    df['Decision'] = df['HoP'] + df['Deputy HoP']
    # change the decision column to 1 if the value is 2 and 0 otherwise
    df['Decision'] = df['Decision'].apply(lambda x: 1 if x == 2 else 0)
    # this means we want those applicants where both HoP and Deputy HoP have accepted
    # change the values of plusminus columns to 2, 1, 0, -1
    df['Grade (++, +, -, 0)'] = df['Grade (++, +, -, 0)'].apply(change_plusminus_value)
    df['Experience (++, +, -, 0)'] = df['Experience (++, +, -, 0)'].apply(change_plusminus_value)
    return df

def compare_grade_experience(grade_score, experience_score):
    if grade_score + experience_score < 2: # if one score is 0, the other must be ++ (2), so the sum is at least + (1)
        return 0
    return 1 # the default state is to not reject the application

def no_negative_score(score):
    if score < 0:
        return 0
    return 1

def rule_checker(rule):
    # this function takes a rule and asks the user to accept or reject it
    # the function returns True as the first value to note the rule has been checked
    # if the user accepts the rule, the function returns True as the second value
    # if the user rejects the rule, the function returns False as the second value
    result = input_cleaner(rule)
    grade_rule_check = True
    if result == 'A':
        print("Thank you for verifying the rule. The rule will be applied.")
        validity = True
    else:
        print("This rule will not be applied. Please note that you must edit the rules in the code if you do not want to apply this rule in the future.")
        validity = False
    return grade_rule_check, validity