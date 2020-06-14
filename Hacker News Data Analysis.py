import os
import re
import pandas as pd
from nltk.tokenize import word_tokenize


# variables
smoothed = 0.5
varColToSplit = ''
list_model = ['wordIn_Title', 'word_Frequencies', 'Post Type', 'smoothed_Probabilities']
probability_type = ''

# setting the path of CSV
my_path = os.getcwd()
inputShpFile = my_path + "\hns_2018_2019.csv"

# reading the CSV and save it as DataFrame
df_HackerNewsDataset = pd.read_csv(inputShpFile)

# assigning the index for the data frame and lowerCase the Title column
new_columns = df_HackerNewsDataset.columns.values.tolist()
new_columns[0] = 'OrginalIndex'
df_HackerNewsDataset.columns = new_columns
df_HackerNewsDataset.set_index("OrginalIndex", inplace=True)
df_HackerNewsDataset['Title'].str.lower()

# Your code will parse the
# files in the training set and build a vocabulary with
# all the words it contains in Title which is Created At 2018.
df_trainingSet2018 = df_HackerNewsDataset[df_HackerNewsDataset.year == 2018]
df_Set2019 = df_HackerNewsDataset[df_HackerNewsDataset.year == 2019]
df_Title208 = df_trainingSet2018.Title.str.lower()


# function to print dataFrame row by row
def df_print(df_arg):
    row_number = 1
    for index, row in df_arg.iterrows():
        row_length = len(row)
        print(row_number, end="  ")
        for rowIndex in range(row_length):
            if rowIndex == row_length - 1:
                print(row[rowIndex])
            else:
                print(row[rowIndex], end="  ")
        row_number += 1


# split a string in cell and expand the data frame by a row for each word in the string
def df_strSplitInRows_toDic(df_arg, colName):
    varColToSplit = colName #TODO
    dict_colValues = {k: [] for k in df_arg.columns.values.tolist()}
    dict_colValues['wordIn_' + colName] = []
    allColNames = df_arg.columns.values.tolist() + ['wordIn_' + colName]
    for row in df_arg.iterrows():
        tokens = word_tokenize(row[1][colName])
        # remove all tokens that are not alphabetic
        vocabularies = [word for word in tokens if word.isalpha()]
        rowValues = []
        for i in range(len(df_arg.columns.values.tolist())):
            rowValues.append(row[1][i])
        for d in vocabularies:
            countRowValues = 1
            for c in allColNames:
                if len(allColNames) > countRowValues:
                    dict_colValues[c].append(rowValues[countRowValues - 1])
                    countRowValues += 1
            dict_colValues['wordIn_' + colName].append(d)
    return dict_colValues


def df_to_txtFile(df_arg):
    file = open(my_path + '\model-2018.txt', "w")
    row_number = 1
    for index, row in df_arg.iterrows():
        row_length = len(row)
        file.write(str(row_number))
        file.write("  ")
        for rowIndex in range(row_length):
            if rowIndex == row_length - 1:
                file.write(str(row[rowIndex]))
                file.write("\n")
            else:
                file.write(str(row[rowIndex]))
                file.write("  ")
        row_number += 1
    df_print(df_arg)


def df_to_txtFile_model(df_arg):
    df_model = df_arg[['wordIn_Title', 'word_Frequencies', 'Post Type', 'smoothed_Probabilities']].copy()
    # sorting
    df_model.drop_duplicates(inplace=True)
    df_model.sort_values(['wordIn_Title', 'Post Type'], inplace=True)
    df_model.drop_duplicates().reset_index(drop=True)
    df_model.index = df_model.index + 1
    pivot_df_model = df_model.pivot_table(index='wordIn_Title',
                                          columns='Post Type',
                                          values='smoothed_Probabilities',
                                          fill_value=0.0)
    print(pivot_df_model)
    file = open(my_path + '\model-2018.txt', "w", encoding='utf-8')
    row_number = 1
    currentWord = ''
    for indexWord, row in pivot_df_model.iterrows():
        if row_number != 1:
            file.write("\n")
        file.write(str(row_number))
        row_number += 1
        file.write("  ")
        file.write(indexWord)
        for postType in row:
            file.write("  ")
            file.write(str(postType))



"""
1. A line counter i, followed by 2 spaces.
2. The word wi, followed by 2 spaces.

3. The frequency of wi in the class story, followed by 2 spaces.
4. The smoothed conditional probability of wi in story – P(wi|story), followed by 2 spaces.
5. The frequency of wi in the class ask_hn, followed by 2 spaces.
6. The smoothed conditional probability of wi in ask_hn – P(wi|ask_hn), followed by 2 spaces.
7. The frequency of wi in the class show_hn, followed by 2 spaces.
8. The smoothed conditional probability of wi in show_hn – P(wi|show_hn), followed by 2 spaces.
9. The frequency of wi in the class poll, followed by 2 spaces.
10. The smoothed conditional probability of wi in poll – P(wi|poll), followed by a carriage
return.
"""


def df_createModel(df_arg, listColNames):
    df_new = df_arg[listColNames]
    df_print(df_new)
    return


# splitting the Title and expanding the array
df_Expanded_TrainingSet2018 = pd.DataFrame(df_strSplitInRows_toDic(df_trainingSet2018, 'Title'))

# Adding column to data frame mapped to frequencies to each word
df_Expanded_TrainingSet2018['word_Frequencies'] = df_Expanded_TrainingSet2018['wordIn_Title'].map(
    df_Expanded_TrainingSet2018['wordIn_Title'].value_counts())

# Adding column to data frame mapped to frequencies to each Post Type
df_Expanded_TrainingSet2018['Post_Type_Frequencies'] = df_Expanded_TrainingSet2018['Post Type'].map(
    df_Expanded_TrainingSet2018['Post Type'].value_counts())

# Calculating the Sum of word frequencies for calculating each word Probability
word_frequencies_sum = df_Expanded_TrainingSet2018['word_Frequencies'].sum()

# Calculating the Probability of each word
df_Expanded_TrainingSet2018['word_Probabilities'] = df_Expanded_TrainingSet2018['word_Frequencies'].apply(
    lambda x: x / word_frequencies_sum)

# Calculating the Sum of Post Type frequencies for calculating each Post Type Probability
Post_Type_frequencies_sum = df_Expanded_TrainingSet2018['Post_Type_Frequencies'].sum()

# Calculating the Probability of each Post Type
df_Expanded_TrainingSet2018['Post_Type_Probabilities'] = df_Expanded_TrainingSet2018['Post_Type_Frequencies'].apply(
    lambda x: x / Post_Type_frequencies_sum)

# Calculating P(Post_type, word)
df_Expanded_TrainingSet2018['P(Post_type, word)'] = df_Expanded_TrainingSet2018['word_Probabilities'] * \
                                                    df_Expanded_TrainingSet2018['Post_Type_Probabilities']

df_Expanded_TrainingSet2018['Post_Type_Probabilities'] = df_Expanded_TrainingSet2018['Post_Type_Frequencies'].apply(
    lambda x: x / Post_Type_frequencies_sum)

# Calculating P(word | Post_type) = (P(Post_type, word) + smoothing Value)/ P(Post Type)
df_Expanded_TrainingSet2018['smoothed_Probabilities'] = (df_Expanded_TrainingSet2018['P(Post_type, word)'] + smoothed) / \
                                                        df_Expanded_TrainingSet2018['Post_Type_Probabilities']

# sorting
df_Expanded_TrainingSet2018.sort_values(['Post Type', 'wordIn_Title'], inplace=True)

# creating table of words Probabilities
df_all_words_Stats_2018 = df_Expanded_TrainingSet2018[
    ['wordIn_Title', 'word_Frequencies', 'word_Probabilities']].copy().drop_duplicates().reset_index(drop=True)
df_all_words_Stats_2018.index = df_all_words_Stats_2018.index + 1

# creating table of Post Type Probabilities
df_Post_Type_Stats_2018 = df_Expanded_TrainingSet2018[
    ['Post Type', 'Post_Type_Frequencies', 'Post_Type_Probabilities']].copy().drop_duplicates().reset_index(drop=True)
df_Post_Type_Stats_2018.index = df_Post_Type_Stats_2018.index + 1

df_to_txtFile_model(df_Expanded_TrainingSet2018)
