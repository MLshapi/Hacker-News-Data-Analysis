import operator
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

# variables
my_path = os.getcwd()
smoothed = 0.5
dict_Models = {}
accuracies = []

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


def df_to_txtFile_model(df_arg, txtFile_arg):
    txtFile_arg.truncate(0)
    file_vocabulary = open(my_path + '\\vocabulary-' + trainYear + '.txt', "w", encoding='utf-8')
    vocabularySetSize = len(df_arg.index)
    row_number = 1
    for word, row in df_arg.iterrows():
        if row_number != 1:
            txtFile_arg.write("\n")
        txtFile_arg.write(str(row_number))
        row_number += 1
        txtFile_arg.write("  ")
        txtFile_arg.write(word)
        file_vocabulary.write(word)
        file_vocabulary.write("\n")
        index_PostType = 0
        for postTypeAndWordFreq in row:
            txtFile_arg.write("  ")
            txtFile_arg.write(str(postTypeAndWordFreq))
            txtFile_arg.write("  ")
            prob = ((postTypeAndWordFreq + smoothed) / (postType_Freq[index_PostType] + vocabularySetSize))
            txtFile_arg.write(str("{:.7f}".format(prob)))
            index_PostType += 1


def createModel(nameOfModel_arg):
    model = pd.read_csv(my_path + "\\" + nameOfModel_arg + ".txt", sep="  ", engine='python', header=None)
    model = model[[x for x in model.columns if x % 2 != 0]]
    model.columns = ['word'] + postTypeOrder
    model.set_index('word', inplace=True)
    dict_Models[nameOfModel_arg] = model
    return model


def df_tester(df_arg, model_arg, resultsFile_arg):
    wrongPrediction = 0
    rightPrediction = 0
    rowNum = 1
    for row in df_arg.iterrows():
        result = TitleTypeFinder(row[1].Title, row[1]['Post Type'], model_arg, rowNum, resultsFile_arg)
        rowNum += 1
        if result == True:
            rightPrediction += 1
        else:
            wrongPrediction += 1

    print("There are ", rightPrediction, "Right Predictions!")
    print("There are ", wrongPrediction, "wrong Predictions!")
    currentAccuracy = rightPrediction / (rightPrediction + wrongPrediction) * 100
    accuracies.append(currentAccuracy)
    print("The acuracy of the Model is ", currentAccuracy, "%")


def TitleTypeFinder(sentence_arg, actualType, model_arg, rowNum, resultsFile_arg):
    tokens = word_tokenize(sentence_arg)
    dict_results = {typeScore: 0 for typeScore in postTypeOrder}
    for word in tokens:
        if word in model_arg.index:
            for type in postTypeOrder:
                x = model_arg.loc[word, type]
                dict_results[type] = dict_results[type] + x

    partialReport = ''
    for x in dict_results.items():
        partialReport = partialReport + "  " + str(x[1])

    probableType = max(dict_results.items(), key=operator.itemgetter(1))[0]

    if actualType == probableType:
        fullReport = str(
            rowNum) + "  " + sentence_arg + "  " + probableType + "  " + partialReport + "  " + actualType + "  Right"
        resultsFile_arg.write(fullReport)
        resultsFile_arg.write("\n")
        return True
    else:
        fullReport = str(
            rowNum) + "  " + sentence_arg + "  " + probableType + "  " + partialReport + "  " + actualType + "  Wrong"
        resultsFile_arg.write(fullReport)
        resultsFile_arg.write("\n")
        return False


def choosingModel():
    print("choose a Model to do do an Experiment on :")
    rowNum = 1
    for m in dict_Models.keys():
        print(rowNum, " ", m)
        rowNum += 1
    userInput = int(input())
    counter = 0
    for m in dict_Models.keys():
        if counter == userInput - 1:
            return dict_Models[m]
        counter += 1


if __name__ == '__main__':
    # splitting the Title and expanding the array
    df_Expanded_TrainingSet2018 = pd.DataFrame(
        df_strSplitInRows_toDic(df_trainingSet2018, 'Title'))

    # opening files to save model & vocabularies & removed words & results
    trainYear = str(df_Expanded_TrainingSet2018.year[0])
    testYear = str(df_Set2019.year.iloc[1])
    file_baselineModel = open(my_path + '\\baselineModel-' + trainYear + '.txt', "w", encoding='utf-8')
    file_removedWords = open(my_path + '\\removedWords-' + trainYear + '.txt', "w", encoding='utf-8')
    file_baselineResult = open(my_path + '\\baseline-result-' + testYear + '.txt', "w", encoding='utf-8')
    file_stopwordModel = open(my_path + '\\stopWord-model-' + trainYear + '.txt', "w", encoding='utf-8')
    file_stopWordResult = open(my_path + '\\stopWord-result-' + testYear + '.txt', "w", encoding='utf-8')
    file_wordlengthModel = open(my_path + '\\wordlength-Model-' + trainYear + '.txt', "w", encoding='utf-8')
    file_wordlengthResult = open(my_path + '\\wordlength-Result-' + testYear + '.txt', "w", encoding='utf-8')
    file_frequentFilteringModel = open(my_path + '\\frequentFiltering-Model-' + trainYear + '.txt', "w",
                                       encoding='utf-8')
    file_frequentFilteringResult = open(my_path + '\\frequentFiltering-Result-' + testYear + '.txt', "w",
                                        encoding='utf-8')
    file_vocabulary = open(my_path + '\\vocabulary-' + trainYear + '.txt', "w", encoding='utf-8')

    # Adding column to data frame mapped to frequencies to each word
    df_Expanded_TrainingSet2018['word_Frequencies'] = df_Expanded_TrainingSet2018['wordIn_Title'].map(
        df_Expanded_TrainingSet2018['wordIn_Title'].value_counts())

    # Adding column to data frame mapped to frequencies to each Post Type
    df_Expanded_TrainingSet2018['Post_Type_Frequencies'] = df_Expanded_TrainingSet2018['Post Type'].map(
        df_Expanded_TrainingSet2018['Post Type'].value_counts())

    # sorting
    df_Expanded_TrainingSet2018.sort_values(['wordIn_Title', 'Post Type'], inplace=True)
    # creating table of Post Type Probabilities

    baseline_model = df_Expanded_TrainingSet2018[['wordIn_Title', 'word_Frequencies', 'Post Type']].copy().pivot_table(
        index='wordIn_Title',
        columns='Post Type',
        aggfunc=len,
        fill_value=0.0
    )

    indexToRemove = baseline_model[baseline_model.max(axis=1) == baseline_model.min(axis=1)]
    for wordToRemove, row in indexToRemove.iterrows():
        file_removedWords.write(wordToRemove)
        file_removedWords.write("\n")

    postTypeOrder = [colName[1] for colName in baseline_model.columns]
    postType_Freq = df_Expanded_TrainingSet2018['Post_Type_Frequencies'].to_dict()
    postType_Freq_total = sum(postType_Freq)
    print(indexToRemove.index)
    # Delete rows that have similar
    baseline_model = baseline_model.drop(indexToRemove.index)

    df_to_txtFile_model(baseline_model, file_baselineModel)

    model_2018 = createModel("model-2018")

    df_tester(df_Set2019, model_2018, file_baselineResult)

    list_stopWords = pd.read_csv(my_path + "\\Stop Words.txt", sep="\n", engine='python', header=None).set_index(
        0).index
    f = open('vocabulary-2018.txt', 'r+')
    list_vocabularies = [word for word in f.read().splitlines()]
    f.close()
    f = open('removedWords-2018.txt', 'r+')
    list_removedWords = [word for word in f.read().splitlines()]
    f.close()

    print("\n")
    print("Experiments Time ...")

    while True:
        ans = input('Do you want remove Stop words? (Y/N)')
        if ans == '' or not ans[0].lower() in ['y', 'n']:
            print('Please answer with yes or no!')
        else:
            break

    list_removedWords_new = list_removedWords.copy()
    if ans[0].lower() == 'y':
        newModel = baseline_model.copy()
        for word in baseline_model.index:
            if word in list_stopWords:
                newModel = newModel.drop(word)
                list_removedWords_new.append(word)
        df_to_txtFile_model(newModel, file_stopwordModel)
        newModel = createModel("stopWord-model-2018")
        df_tester(df_Set2019, newModel, file_stopWordResult)
        file_removedWordsE1 = open(my_path + '\\removedWords-stopWords-' + trainYear + '.txt', "w", encoding='utf-8')
        for word in list_removedWords_new:
            file_removedWords.write(word)
            file_removedWords.write(" ")

    while True:
        ans = input('Do you want remove words with length not between 3 AND 9? (Y/N)')
        if ans == '' or not ans[0].lower() in ['y', 'n']:
            print('Please answer with yes or no!')
        else:
            break

    list_removedWords_new = list_removedWords.copy()
    if ans[0].lower() == 'y':
        newModel = baseline_model.copy()
        indexToRemove = [word for word in newModel.index if len(word) <= 2 or len(word) >= 9]
        newModel = newModel.drop(indexToRemove)
        df_to_txtFile_model(newModel, file_wordlengthModel)
        newModel = createModel("wordlength-Model-2018")
        df_tester(df_Set2019, newModel, file_baselineResult)

    while True:
        ans = input('Do you want try E3? (Y/N)')
        if ans == '' or not ans[0].lower() in ['y', 'n']:
            print('Please answer with yes or no!')
        else:
            break

    if ans[0].lower() == 'y':
        arrFreq = [1, 5, 10, 15, 20]
        accuracies = []
        for num in arrFreq:
            list_removedWords_new = list_removedWords.copy()
            newModel = baseline_model.copy()
            indexToRemove = newModel[baseline_model.sum(axis=1) <= num]
            newModel = newModel.drop(indexToRemove.index)
            for wordToRemove, row in indexToRemove.iterrows():
                list_removedWords_new.append(wordToRemove)
            df_to_txtFile_model(newModel, file_frequentFilteringModel)
            newModel = createModel("frequentFiltering-Model-2018")
            df_tester(df_Set2019, newModel, file_frequentFilteringResult)
            print('\n')
        # this is for plotting purpose
        arrLabels = ['1<freq', '5<freq', '10<freq', '15<freq', '20<freq']
        dict_toPlot = {"accuracies": accuracies, "Labels": arrLabels}
        df = pd.DataFrame(dict_toPlot, index=arrLabels).plot(kind='bar')
        plt.show()

        baseline_model_sorted = baseline_model.copy()
        baseline_model_sorted['sum_cols'] = baseline_model_sorted.sum(axis=1)
        baseline_model_sorted = baseline_model_sorted.sort_values('sum_cols', ascending=False)
        baseline_model_sorted = baseline_model_sorted.drop(['sum_cols'], axis=1)

        arrFreqPercentages = [5, 10, 15, 20, 25]
        accuracies = []
        for num in arrFreqPercentages:
            list_removedWords_new = list_removedWords.copy()
            numOfRows = len(baseline_model.index) - int((len(baseline_model.index)*num)/100)
            indexToRemove = baseline_model_sorted.head(int((len(baseline_model.index)*num)/100)).index
            newModel = baseline_model_sorted.tail(numOfRows)
            # for wordToRemove, row in indexToRemove.iterrows():
            #     list_removedWords_new.append(wordToRemove)
            df_to_txtFile_model(newModel, file_frequentFilteringModel)
            newModel = createModel("frequentFiltering-Model-2018")
            df_tester(df_Set2019, newModel, file_frequentFilteringResult)
            print('\n')
        # this is for plotting purpose
        arrLabels = ['5%', '10%', '15%', '20%', '25%']
        dict_toPlot = {"accuracies": accuracies, "Labels": arrLabels}
        df = pd.DataFrame(dict_toPlot, index=arrLabels).plot(kind='bar')
        plt.show()
