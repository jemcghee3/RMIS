import pandas as pd
import re

def experience_calculator(experience):
    # this function takes a string of experience and returns a number of months
    if experience == None:
        return 0
    if type(experience) == float:
        return 0 # this is a NaN
    if len(experience) == 0:
        return 0
    exp_regex = re.compile(r'(\d+)\s*[y]\s*(\d+[m])*')
    mo = exp_regex.search(experience)
    if mo == None:
        return 0
    if mo.group(1) != None:
        years = int(mo.group(1))
    else:
        years = 0
    if mo.group(2) != None:
        months = int(mo.group(2)[:-1])
    else:
        months = 0
    months = months + years * 12
    return months

def experience_converter(months):
    # takes a string of months and returns a +/- value
    months = int(months)
    if months < 6:
        return '0'
    elif months < 24:
        return '+'
    elif months >= 24:
        return '++'
    else:
        return '-'

def grade_calculator(grade):
    # this function takes a string of grade and returns a GPA
#    print("input grade " + str(grade))
    if grade == None:
        return 0
    grade = str(grade)
    if len(grade) == 0:
        return 0
    exp_regex = re.compile(r'(\d)(\.*\d*)')
    mo = exp_regex.search(grade)
#    print("mo is " + str(mo))
    if mo == None:
        return 0
    if mo.group(1) != None:
        first = int(mo.group(1))
    else:
        first = 0
    if mo.group(2) != None and len(mo.group(2)) > 0:
        second = int(mo.group(2)[1:])
    else:
        second = 0
    GPA = str(first) + '.' + str(second)
    print("output gpa " + str(GPA))
    return GPA

def grade_converter(grade):
    # takes a string of grade and returns a +/- value
    grade = float(grade)
    print("input grade " + str(grade))
    if grade >= 5.3:
        print("output grade " + '++')
        return '++'
    elif grade >= 4.8:
        print("output grade " + '+')
        return '+'
    elif grade >= 4.6:
        print("output grade " + '0')
        return '0'
    else:
        print("output grade " + '-')
        return '-'

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
    elif value == '0':
        return 0
    else:
#        print("Error: the value is not valid")
        return None
    
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
    # drop the rows with NaN values in the decision column, experience column, and grade column
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
    df['Experience (y,m)'] = df['Experience (y,m)'].apply(experience_calculator)
    df['Experience (++, +, -, 0)'] = df['Experience (y,m)'].apply(experience_converter)
    df['Grade'] = df['Grade'].apply(grade_calculator)
    df['Grade (++, +, -, 0)'] = df['Grade'].apply(grade_converter)
    df['Grade (++, +, -, 0)'] = df['Grade (++, +, -, 0)'].apply(change_plusminus_value)
    df['Experience (++, +, -, 0)'] = df['Experience (++, +, -, 0)'].apply(change_plusminus_value)
    df = df.dropna(subset=['Grade (++, +, -, 0)'])
    df = df.dropna(subset=['Experience (++, +, -, 0)'])
    return df

def compare_grade_experience(grade_score, experience_score):
    if grade_score == 0 and experience_score != 2: 
        return 0
    elif grade_score != 2 and experience_score == 0: 
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