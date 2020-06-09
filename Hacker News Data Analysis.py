import os
import re
import pandas as pd

regex = re.compile('[^a-zA-Z0-9\s]')

# setting the path of CSV
my_path = os.getcwd()
inputShpFile = my_path + "\hns_2018_2019.csv"

# reading the CSV and save it as DataFrame
df_HackerNewsDataset = pd.read_csv(inputShpFile)

# assigning the index for the data frame and lowerCase the Title column
new_columns = df_HackerNewsDataset.columns.values.tolist()
new_columns[0] = 'index'
df_HackerNewsDataset.columns = new_columns
df_HackerNewsDataset.set_index("index", inplace=True)
df_HackerNewsDataset['Title'].str.lower()

# Your code will parse the
# files in the training set and build a vocabulary with
# all the words it contains in Title which is Created At 2018.
df_trainingSet2018 = df_HackerNewsDataset[df_HackerNewsDataset.year == 2018]
df_Set2019 = df_HackerNewsDataset[df_HackerNewsDataset.year == 2019]
df_Title208 = df_trainingSet2018.Title.str.lower()


def expand_df_ByVocabularyInCol(df_toExpand):
    df_toExpand_copy = df_toExpand.copy()
    df_toExpand_copy["VocabularyInTitle"] = "empty"
    df_Expanded = pd.DataFrame(columns=df_toExpand_copy.columns.values.tolist())
    for index, row in df_toExpand_copy.iterrows():
        vocabulariesInRow = row["Title"].split()
        for v in vocabulariesInRow:
            row["VocabularyInTitle"] = v
            df_Expanded = df_Expanded.append(row, ignore_index=True)
    return df_Expanded


# expanding
def expander(df_toParse):
    dict_vocabulary = {k: [] for k in df_HackerNewsDataset.columns.values.tolist()}
    dict_vocabulary['TitleIndex'] = []
    dict_vocabulary['allWordsInTitle'] = []
    count_TitleIndex = 1
    for row in df_toParse.iterrows():
        vocabulariesInRow = regex.sub('', row[1].Title).split()
        rowPostType = row[1]["Post Type"]
        rowTitle = row[1]["Title"]
        rowObjectID = row[1]["Object ID"]
        rowAuthor = row[1]["Author"]
        rowCreatedAt = row[1]["Created At"]
        rowURL = row[1]["URL"]
        rowPoints = row[1]["Points"]
        rowNumberofComments = row[1]["Number of Comments"]
        rowYear = row[1]["year"]
        for v in vocabulariesInRow:
            dict_vocabulary['TitleIndex'].append(count_TitleIndex)
            dict_vocabulary["allWordsInTitle"].append(v)
            dict_vocabulary["Post Type"].append(rowPostType)
            dict_vocabulary["Title"].append(rowTitle)
            dict_vocabulary["Object ID"].append(rowObjectID)
            dict_vocabulary["Author"].append(rowAuthor)
            dict_vocabulary["Created At"].append(rowCreatedAt)
            dict_vocabulary["URL"].append(rowURL)
            dict_vocabulary["Points"].append(rowPoints)
            dict_vocabulary["Number of Comments"].append(rowNumberofComments)
            dict_vocabulary["year"].append(rowYear)
        count_TitleIndex += 1
    return dict_vocabulary


df_Expanded_TrainingSet2018 = pd.DataFrame(expander(df_trainingSet2018))
print(df_Expanded_TrainingSet2018['Post Type'].str.get_dummies().T.dot(pd.get_dummies(df_Expanded_TrainingSet2018["allWordsInTitle"])))
df_trainingSet2018_stats = df_Expanded_TrainingSet2018.groupby(['Post Type']).count()['allWordsInTitle']
#df_trainingSet2018_stats = df_Expanded_TrainingSet2018['Post Type'].value_counts().to_frame()
print(df_trainingSet2018_stats)
print(g_df['allWordsInTitle'].sum/len(df_Expanded_TrainingSet2018.index) for gName,g_df in df_trainingSet2018_stats)
df_trainingSet2018_stats["percentages"] = [g_df['allWordsInTitle'].sum/len(df_Expanded_TrainingSet2018.index) for gName,g_df in df_trainingSet2018_stats]


