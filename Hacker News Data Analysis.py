# -------------------------------------------------------
# Assignment (2)
# Written by (Moayad ALshapi and student id: 40037861)
# For COMP 472 Section (IX) – Summer 2020
# --------------------------------------------------------
import math
import operator
import os

import matplotlib.pyplot as plt
import pandas as pd
from nltk.tokenize import word_tokenize

# variables
my_path = os.getcwd()
smoothed = 0.5
accuracies = []

# setting the path of CSV
inputShpFile = my_path + "\hns_2018_2019.csv"

# reading the CSV and save it as DataFrame
df_HackerNewsDataset = pd.read_csv(inputShpFile)

# assigning the index for the data frame and lowerCase the Title column
new_columns = df_HackerNewsDataset.columns.values.tolist()
df_HackerNewsDataset['Title'] = df_HackerNewsDataset['Title'].str.lower()

df_HackerNewsDataset['Created AT Year'] = pd.DatetimeIndex(df_HackerNewsDataset['Created At']).year
years = df_HackerNewsDataset['Created AT Year'].drop_duplicates().values

# splitting the data by year for training and testing purposes
df_trainingSet2018 = df_HackerNewsDataset[df_HackerNewsDataset.year == years.min()]
df_trainingSet2018 = df_trainingSet2018[['Title', 'Post Type', 'year']]
df_Set2019 = df_HackerNewsDataset[df_HackerNewsDataset.year == years.max()]
df_Set2019 = df_Set2019[['Title', 'Post Type', 'year']]


# expanding Dataframe by splitting string column
def df_strSplitInRows_toDic(df_arg, colName):
    # making a dictionary of columns names with empty arrays
    dict_colValues = {k: [] for k in df_arg.columns.values.tolist()}
    # adding item to the dictionary to contains the words of string that we are going to split
    dict_colValues['wordIn_' + colName] = []
    allColNames = df_arg.columns.values.tolist() + ['wordIn_' + colName]
    # looping through the rows in df_arg and filling the dict_colValues
    for currentRow in df_arg.iterrows():
        tokens = word_tokenize(currentRow[1][colName])
        # remove all tokens that are not alphabetic
        vocabularies = [word for word in tokens if word.isalpha()]
        rowValues = []
        for i in range(len(df_arg.columns.values.tolist())):
            rowValues.append(currentRow[1][i])
        for d in vocabularies:
            countRowValues = 1
            for c in allColNames:
                if len(allColNames) > countRowValues:
                    dict_colValues[c].append(rowValues[countRowValues - 1])
                    countRowValues += 1
            dict_colValues['wordIn_' + colName].append(d)
    return dict_colValues


# wring the model into a txt file & creating vocabulary txt file
def df_to_txtFile_model(df_arg, txtFile_arg):
    txtFile_arg.truncate(0)
    file_vocabulary = open(my_path + '\\vocabulary-' + trainYear + '.txt', "w", encoding='utf-8')
    vocabularySetSize = len(df_arg.index)
    row_number = 1
    for word, currentRow in df_arg.iterrows():
        if row_number != 1:
            txtFile_arg.write("\n")
        txtFile_arg.write(str(row_number))
        row_number += 1
        txtFile_arg.write("  ")
        txtFile_arg.write(word)
        # creating the vocabulary list txt
        file_vocabulary.write(word)
        file_vocabulary.write("\n")
        index_PostType = 0
        for postTypeAndWordFreq in currentRow:
            txtFile_arg.write("  ")
            txtFile_arg.write(str(postTypeAndWordFreq))
            txtFile_arg.write("  ")
            prob = ((postTypeAndWordFreq + smoothed) / (
                    postType_Freq.iloc[index_PostType] + smoothed * vocabularySetSize))
            txtFile_arg.write(str("{:.7f}".format(prob)))
            index_PostType += 1
    file_vocabulary.flush()
    txtFile_arg.flush()


# Function to Create the model
def createModel(nameOfModel_arg):
    model = pd.read_csv(my_path + "\\" + nameOfModel_arg + ".txt", sep="  ", engine='python', header=None)
    model = model[[col for col in model.columns if col % 2 != 0]]
    model.columns = ['word'] + postTypeOrder
    model.set_index('word', inplace=True)
    return model


# function to test the model with set of data
def df_tester(df_arg, model_arg, resultsFile_arg):
    resultsFile_arg.truncate(0)
    wrongPrediction = 0
    rightPrediction = 0
    rowNum = 1
    for rowIn in df_arg.iterrows():
        result = TitleTypeFinder(rowIn[1].Title, rowIn[1]['Post Type'], model_arg, rowNum, resultsFile_arg)
        rowNum += 1
        if result:
            rightPrediction += 1
        else:
            wrongPrediction += 1
    # printing some stats to show the accuracy of the model
    print("There are ", rightPrediction, "Right Predictions!")
    print("There are ", wrongPrediction, "wrong Predictions!")
    currentAccuracy = rightPrediction / (rightPrediction + wrongPrediction) * 100
    # storing the result for comparision purpose
    accuracies.append(currentAccuracy)
    print("The acuracy of the Model is ", currentAccuracy, "%")


# to predict the type of the title
def TitleTypeFinder(sentence_arg, actualType, model_arg, rowNum, resultsFile_arg):
    tokens = word_tokenize(sentence_arg)
    # dictionary to stores the scores of each words in order to predict the most likely postType
    dict_results = {typeScore: 0 for typeScore in postTypeOrder}
    for word in tokens:
        # check if the word is in model and it is also means it is in vocabulary
        if word in model_arg.index:
            for postType in postTypeOrder:
                prob = model_arg.loc[word, postType]
                dict_results[postType] = dict_results[postType] + math.log10(prob)

    partialReport = ''
    for res in dict_results.items():
        partialReport = partialReport + "  " + str(res[1])

    # picking the maximum Value
    probableType = max(dict_results.items(), key=operator.itemgetter(1))[0]

    # writing the report with the predictions
    if actualType == probableType:
        fullReport = str(
            rowNum) + "  " + sentence_arg + "  " + probableType + "  " + partialReport + "  " + actualType + "  Right"
        resultsFile_arg.write(fullReport)
        resultsFile_arg.write("\n")
        resultsFile_arg.flush()
        return True
    else:
        fullReport = str(
            rowNum) + "  " + sentence_arg + "  " + probableType + "  " + partialReport + "  " + actualType + "  Wrong"
        resultsFile_arg.write(fullReport)
        resultsFile_arg.write("\n")
        resultsFile_arg.flush()
        return False


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

    # Adding column to data frame mapped to frequencies to each word
    df_Expanded_TrainingSet2018['word_Frequencies'] = df_Expanded_TrainingSet2018['wordIn_Title'].map(
        df_Expanded_TrainingSet2018['wordIn_Title'].value_counts())

    # Adding column to data frame mapped to frequencies to each Post Type
    df_Expanded_TrainingSet2018['Post_Type_Frequencies'] = df_Expanded_TrainingSet2018['Post Type'].map(
        df_Expanded_TrainingSet2018['Post Type'].value_counts())

    # sorting
    df_Expanded_TrainingSet2018.sort_values(['wordIn_Title', 'Post Type'], inplace=True)

    # creating pivot table of Post Type Probabilities which helps to know the frequency of each word in each postType
    baseline_model = df_Expanded_TrainingSet2018[['wordIn_Title', 'word_Frequencies', 'Post Type']].copy().pivot_table(
        index='wordIn_Title',
        columns='Post Type',
        aggfunc=len,
        fill_value=0.0
    )

    # removing the words that has equal frequency in each type because it dose not gives any indication about postType
    indexToRemove = baseline_model[baseline_model.max(axis=1) == baseline_model.min(axis=1)]
    for wordToRemove, row in indexToRemove.iterrows():
        file_removedWords.write(wordToRemove)
        file_removedWords.write("\n")
        file_removedWords.flush()
    baseline_model = baseline_model.drop(indexToRemove.index)

    # creating some Variable to ease the coding process
    postTypeOrder = [colName[1] for colName in baseline_model.columns]
    postType_Freq = df_Expanded_TrainingSet2018['Post_Type_Frequencies'].drop_duplicates()
    postType_Freq_total = sum(postType_Freq)

    # creating the baseline txt file
    df_to_txtFile_model(baseline_model, file_baselineModel)

    # creating the dataFrame model of baseline to use it in the prediction process
    baseLine_model_2018 = createModel("baselineModel-2018")

    # testing the model with testing data
    df_tester(df_Set2019, baseLine_model_2018, file_baselineResult)

    # creating lists of stopWords & vocabularies & removed words for convenience
    list_stopWords = pd.read_csv(my_path + "\\Stop Words.txt", sep="\n", engine='python', header=None).set_index(
        0).index
    f = open('vocabulary-2018.txt', 'r+')
    list_vocabularies = [word for word in f.read().splitlines()]
    f.close()
    f = open('removedWords-2018.txt', 'r+')
    list_removedWords = [word for word in f.read().splitlines()]
    f.close()

    # beginning of Experiments
    numberOfExperiment = 0
    print("\n")
    print("Experiments Time ...")
    while True:
        print("\n")
        while True:
            print("Press 0 to End..")
            print("Which Experiment would you like to try:")
            print("1) remove Stop words")
            print("2) remove words according to their lengths")
            print("3) show how accuracy changes by removing words according to their frequencies")
            ans = int(input())
            if 0 <= ans <= 3:
                break
            else:
                print("please enter a valid number")

        # End the Code
        if ans == 0:
            print("THE CODE ENDED!..")
            break

        # experiment Number 1 : removing stop words and check if the model gets better
        if ans == 1:
            file_removedWordsAfterExperiment = open(
                my_path + '\\removedWordsExperiment-stopWords' + testYear + '.txt', "w", encoding='utf-8')
            list_removedWords_new = list_removedWords.copy()
            newModel = baseline_model.copy()
            for word in baseline_model.index:
                if word in list_stopWords:
                    newModel = newModel.drop(word)
                    list_removedWords_new.append(word)
            df_to_txtFile_model(newModel, file_stopwordModel)
            newModel = createModel("stopWord-model-2018")
            df_tester(df_Set2019, newModel, file_stopWordResult)
            # creating removed words list
            for word in list_removedWords_new:
                file_removedWordsAfterExperiment.write(word)
                file_removedWordsAfterExperiment.write("\n")
            # check if the result is better than the baseline model
            if accuracies[0] == accuracies[-1]:
                print("no Changes!")
            elif accuracies[0] < accuracies[-1]:
                print("better model by ", accuracies[-1] - accuracies[0], '%')
            else:
                print("worse model by ", accuracies[0] - accuracies[-1], '%')
            file_removedWordsAfterExperiment.flush()
            file_removedWordsAfterExperiment.close()

        # experiment Number 2 : removing words according to their length and check if the model gets better
        if ans == 2:
            file_removedWordsAfterExperiment = open(
                my_path + '\\removedWordsExperiment-length' + testYear + '.txt', "w", encoding='utf-8')
            print("Enter the length limits:")
            print('Note: according to the assignment requirements minimum is 2 and maximum is 9')
            while True:
                x = int(input("minimum length = "))
                y = int(input("maximum length = "))
                if y <= x:
                    print("please Enter another values")
                else:
                    break
            list_removedWords_new = list_removedWords.copy()
            newModel = baseline_model.copy()
            indexToRemove = [word for word in newModel.index if len(word) <= x or len(word) >= y]
            newModel = newModel.drop(indexToRemove)
            df_to_txtFile_model(newModel, file_wordlengthModel)
            newModel = createModel("wordlength-Model-2018")
            df_tester(df_Set2019, newModel, file_wordlengthResult)
            # creating removed words list
            for word in list_removedWords_new:
                file_removedWordsAfterExperiment.write(word)
                file_removedWordsAfterExperiment.write("\n")
            for word in indexToRemove:
                file_removedWordsAfterExperiment.write(word)
                file_removedWordsAfterExperiment.write("\n")
            # check if the result is better than the baseline model
            if accuracies[0] == accuracies[-1]:
                print("no Changes")
            elif accuracies[0] < accuracies[-1]:
                print("better model by ", accuracies[-1] - accuracies[0], '%')
            else:
                print("worse model by ", accuracies[0] - accuracies[-1], '%')
            file_removedWordsAfterExperiment.flush()
            file_removedWordsAfterExperiment.close()

        # experiment Number 3 : removing words words according their frequencies and plot the results
        if ans == 3:
            newModelSorted = baseline_model.copy()
            # sorting the pivot table for Experiment number 3
            newModelSorted['sum_cols'] = newModelSorted.sum(axis=1)
            newModelSorted = newModelSorted.sort_values('sum_cols', ascending=False).drop(['sum_cols'], axis=1)
            file_removedWordsAfterExperiment = open(
                my_path + '\\removedWordsExperiment-frequencies 1 -' + testYear + '.txt', "w", encoding='utf-8')
            # first part of experiment 3
            arrFreq = [1, 5, 10, 15, 20]
            accuracies = []
            for num in arrFreq:
                newModel = newModelSorted.copy()
                list_removedWords_new = list_removedWords.copy()
                indexToRemove = newModel[newModel.sum(axis=1) <= num]
                newModel = newModel.drop(indexToRemove.index)
                for wordToRemove, row in indexToRemove.iterrows():
                    list_removedWords_new.append(wordToRemove)
                df_to_txtFile_model(newModel, file_frequentFilteringModel)
                newModel = createModel("frequentFiltering-Model-2018")
                df_tester(df_Set2019, newModel, file_frequentFilteringResult)
                print('\n')
            for word in list_removedWords_new:
                file_removedWordsAfterExperiment.write(word)
                file_removedWordsAfterExperiment.write("\n")
            file_removedWordsAfterExperiment.flush()
            file_removedWordsAfterExperiment.close()
            # this is for plotting purpose
            arrLabels = ['1<freq', '5<freq', '10<freq', '15<freq', '20<freq']
            dict_toPlot = {"accuracies": accuracies, "Labels": arrLabels}
            pd.DataFrame(dict_toPlot, index=arrLabels).plot(kind='bar')
            plt.show()

            # second part of experiment 3
            file_removedWordsAfterExperiment = open(
                my_path + '\\removedWordsExperiment-frequencies 2 -' + testYear + '.txt', "w", encoding='utf-8')
            newModel = newModelSorted.copy()
            arrFreqPercentages = [5, 10, 15, 20, 25]
            accuracies = []
            for num in arrFreqPercentages:
                list_removedWords_new = list_removedWords.copy()
                numOfRows = len(baseline_model.index) - int((len(baseline_model.index) * num) / 100)
                indexToRemove = newModelSorted.head(int((len(baseline_model.index) * num) / 100)).index
                newModel = newModelSorted.tail(numOfRows)
                for wordToRemove in indexToRemove:
                    list_removedWords_new.append(wordToRemove)
                df_to_txtFile_model(newModel, file_frequentFilteringModel)
                newModel = createModel("frequentFiltering-Model-2018")
                df_tester(df_Set2019, newModel, file_frequentFilteringResult)
                print('\n')
            for word in list_removedWords_new:
                file_removedWordsAfterExperiment.write(word)
                file_removedWordsAfterExperiment.write("\n")
            file_removedWordsAfterExperiment.flush()
            file_removedWordsAfterExperiment.close()
            # this is for plotting purpose
            arrLabels = ['5%', '10%', '15%', '20%', '25%']
            dict_toPlot = {"accuracies": accuracies, "Labels": arrLabels}
            pd.DataFrame(dict_toPlot, index=arrLabels).plot(kind='bar')
            plt.show()
