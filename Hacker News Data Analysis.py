import os

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
df_HackerNewsDataset['Title'] = df_HackerNewsDataset['Title'].str.lower()
# Your code will parse the
# files in the training set and build a vocabulary with
# all the words it contains in Title which is Created At 2018.
df_trainingSet2018 = df_HackerNewsDataset[df_HackerNewsDataset.year == 2018]
df_Set2019 = df_HackerNewsDataset[df_HackerNewsDataset.year == 2019]
df_Title208 = df_trainingSet2018.Title.str.lower()


# split a string in cell and expand the data frame by a row for each word in the string
def df_strSplitInRows_toDic(df_arg, colName):
    varColToSplit = colName  # TODO
    dict_colValues = {k: [] for k in df_arg.columns.values.tolist()}
    dict_colValues['wordIn_' + colName] = []
    allColNames = df_arg.columns.values.tolist() + ['wordIn_' + colName]
    for row in df_arg.iterrows():
        tokens = word_tokenize(row[1][colName])
        # remove all tokens that are not alphabetic
        # todo make removing words here
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


def df_to_txtFile_model(df_arg , year):
    # creating table of Post Type Probabilities
    df_Post_Type_Stats = df_arg[
        ['Post Type', 'Post_Type_Frequencies']].copy().drop_duplicates().sort_values(
        'Post Type').reset_index(drop=True)
    df_arg.index = df_arg.index + 1
    df_model = df_arg[['wordIn_Title', 'word_Frequencies', 'Post Type']].copy()
    # sorting
    df_model.sort_values(['wordIn_Title', 'Post Type'], inplace=True)
    df_model.reset_index(drop=True)
    df_model.index = df_model.index + 1
    pivot_df_model = df_model.pivot_table(index='wordIn_Title',
                                          columns='Post Type',
                                          aggfunc=len,
                                          fill_value=0.0
                                          )
    file_model = open(my_path + '\\model-'+year+'.txt', "w", encoding='utf-8')
    file_vocabulary = open(my_path + '\\vocabulary-'+year+'.txt', "w", encoding='utf-8')
    file_removedWords = open(my_path + '\\removedWords-'+year+'.txt', "w", encoding='utf-8')

    row_number = 1
    for word, row in pivot_df_model.iterrows():
        if row.max() == row.min() or len(word) < 2:
            file_removedWords.write(word)
            file_removedWords.write("\n")
            continue
        if row_number != 1:
            file_model.write("\n")
        file_model.write(str(row_number))
        row_number += 1
        file_model.write("  ")
        file_model.write(word)
        file_vocabulary.write(word)
        file_vocabulary.write("\n")
        counter_PostType = 0
        for postTypeAndWordFreq in row:
            postType = df_model['Post Type'].drop_duplicates().iloc[counter_PostType]
            counter_PostType += 1
            postTypeFreq = df_Post_Type_Stats[df_Post_Type_Stats["Post Type"].isin([postType])].reset_index(
                drop=True).iloc[0][1]
            file_model.write("  ")
            file_model.write(str(postTypeAndWordFreq))
            file_model.write("  ")
            prob = ((postTypeAndWordFreq + smoothed) / postTypeFreq) * 100
            file_model.write(str(prob))


# splitting the Title and expanding the array
df_Expanded_TrainingSet2018 = pd.DataFrame(df_strSplitInRows_toDic(df_trainingSet2018, 'Title'))

# Adding column to data frame mapped to frequencies to each word
df_Expanded_TrainingSet2018['word_Frequencies'] = df_Expanded_TrainingSet2018['wordIn_Title'].map(
    df_Expanded_TrainingSet2018['wordIn_Title'].value_counts())

# Adding column to data frame mapped to frequencies to each Post Type
df_Expanded_TrainingSet2018['Post_Type_Frequencies'] = df_Expanded_TrainingSet2018['Post Type'].map(
    df_Expanded_TrainingSet2018['Post Type'].value_counts())


df_to_txtFile_model(df_Expanded_TrainingSet2018, str(df_Expanded_TrainingSet2018.year[0]))
