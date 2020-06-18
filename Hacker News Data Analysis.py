import operator
import os
import pandas as pd
from nltk.tokenize import word_tokenize
from decimal import *
getcontext().prec = 6

# variables
my_path = os.getcwd()
smoothed = 0.5
varColToSplit = ''
list_model = ['wordIn_Title', 'word_Frequencies', 'Post Type', 'smoothed_Probabilities']
probability_type = ''
wrongPrediction = 0
rightPrediction = 0

# setting the path of CSV
inputShpFile = my_path + "\hns_2018_2019.csv"

# reading the CSV and save it as DataFrame
df_HackerNewsDataset = pd.read_csv(inputShpFile)

# assigning the index for the data frame and lowerCase the Title column
new_columns = df_HackerNewsDataset.columns.values.tolist()
df_HackerNewsDataset['Title'] = df_HackerNewsDataset['Title'].str.lower()
# Your code will parse the
# files in the training set and build a vocabulary with
# all the words it contains in Title which is Created At 2018.
df_trainingSet2018 = df_HackerNewsDataset[df_HackerNewsDataset.year == 2018]
df_trainingSet2018 = df_trainingSet2018[['Title', 'Post Type', 'year']]
df_Set2019 = df_HackerNewsDataset[df_HackerNewsDataset.year == 2019]
df_Set2019 = df_Set2019[['Title', 'Post Type', 'year']]


# split a string in cell and expand the data frame by a row for each word in the string
def df_strSplitInRows_toDic(df_arg, colName):
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


def df_to_txtFile_model(df_arg):
    vocabularySetSize = len(df_arg.index)
    row_number = 1
    for word, row in df_arg.iterrows():
        if row_number != 1:
            file_model.write("\n")
        file_model.write(str(row_number))
        row_number += 1
        file_model.write("  ")
        file_model.write(word)
        file_vocabulary.write(word)
        file_vocabulary.write("\n")
        index_PostType = 0
        for postTypeAndWordFreq in row:
            file_model.write("  ")
            file_model.write(str(postTypeAndWordFreq))
            file_model.write("  ")
            prob = ((postTypeAndWordFreq + smoothed) / (postType_Freq[index_PostType] + vocabularySetSize))
            file_model.write(str("{:.7f}".format(prob)))
            index_PostType += 1






def df_tester(df_arg):
    wrongPrediction = 0
    rightPrediction = 0

    for row in df_arg.iterrows():
        result = TitleTypeFinder(row[1].Title, row[1]['Post Type'], 1)
        if result == True:
            rightPrediction += 1
        else:
            wrongPrediction += 1

    print("There are ", rightPrediction , "Right Predictions!")
    print("There are ", wrongPrediction, "wrong Predictions!")
    print("The acuracy of the Model is ", (rightPrediction / (rightPrediction + wrongPrediction) * 100), "%")


def TitleTypeFinder(sentence_arg, actualType, rowNum):
    """
    1 Y Combinator story 0.004 0.001 0.0002. 0.002 story right
    2 A Student's Guide poll 0.002 0.03 0.007 0.12 story wrong
    :param sentence_arg:
    :param rowNum:
    :return:
    """
    tokens = word_tokenize(sentence_arg)
    dict_results = {typeScore: 0 for typeScore in postTypeOrder}
    for word in tokens:
        if word in model.index:
            for type in postTypeOrder:
                x = model.loc[word, type]
                dict_results[type] = dict_results[type] + x

    partialReport = ''
    for x in dict_results.items():
        partialReport = partialReport + "  " + str(x[1])

    probableType = max(dict_results.items(), key=operator.itemgetter(1))[0]


    if actualType == probableType:
        fullReport = str(
            rowNum) + "  " + sentence_arg + "  " + probableType + "  " + partialReport + "  " + actualType + "  Right"
        baselineResult.write(fullReport)
        baselineResult.write("\n")
        return True
    else:
        fullReport = str(
            rowNum) + "  " + sentence_arg + "  " + probableType + "  " + partialReport + "  " + actualType + "  Wrong"
        baselineResult.write(fullReport)
        baselineResult.write("\n")
        return False


if __name__ == '__main__':
    # splitting the Title and expanding the array
    df_Expanded_TrainingSet2018 = pd.DataFrame(
        df_strSplitInRows_toDic(df_trainingSet2018, 'Title'))

    # opening files to save model & vocabularies and removed words
    trainYear = str(df_Expanded_TrainingSet2018.year[0])
    testYear = str(df_Set2019.year.iloc[1])
    file_model = open(my_path + '\\model-' + trainYear + '.txt', "w", encoding='utf-8')
    file_vocabulary = open(my_path + '\\vocabulary-' + trainYear + '.txt', "w", encoding='utf-8')
    file_removedWords = open(my_path + '\\removedWords-' + trainYear + '.txt', "w", encoding='utf-8')

    # Adding column to data frame mapped to frequencies to each word
    df_Expanded_TrainingSet2018['word_Frequencies'] = df_Expanded_TrainingSet2018['wordIn_Title'].map(
        df_Expanded_TrainingSet2018['wordIn_Title'].value_counts())

    # Adding column to data frame mapped to frequencies to each Post Type
    df_Expanded_TrainingSet2018['Post_Type_Frequencies'] = df_Expanded_TrainingSet2018['Post Type'].map(
        df_Expanded_TrainingSet2018['Post Type'].value_counts())

    # sorting
    df_Expanded_TrainingSet2018.sort_values(['wordIn_Title', 'Post Type'], inplace=True)
    # creating table of Post Type Probabilities

    pivot_df_model = df_Expanded_TrainingSet2018[['wordIn_Title', 'word_Frequencies', 'Post Type']].copy().pivot_table(
        index='wordIn_Title',
        columns='Post Type',
        aggfunc=len,
        fill_value=0.0
    )

    indexToRemove = pivot_df_model[pivot_df_model.max(axis=1) == pivot_df_model.min(axis=1)]
    for wordToRemove, row in indexToRemove.iterrows():
        file_removedWords.write(wordToRemove)
        file_removedWords.write("\n")

    postTypeOrder = [colName[1] for colName in pivot_df_model.columns]
    postType_Freq = df_Expanded_TrainingSet2018['Post_Type_Frequencies'].to_dict()
    postType_Freq_total = sum(postType_Freq)

    # Delete rows that have similar
    pivot_df_model = pivot_df_model.drop(indexToRemove.index)

    df_to_txtFile_model(pivot_df_model)

    model = pd.read_csv(my_path + "\model-2018.txt", sep="  ", engine='python', header=None)
    model = model[[x for x in model.columns if x % 2 != 0]]
    model.columns = ['word'] + postTypeOrder
    model.set_index('word', inplace=True)

    baselineResult = open(my_path + '\\baseline-result-' + testYear + '.txt', "w", encoding='utf-8')

    df_tester(df_Set2019)
