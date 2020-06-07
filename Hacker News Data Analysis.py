import os

import pandas as pd

my_path = os.getcwd()
inputShpFile = my_path + "\hns_2018_2019.csv"
# reading the CSV and save it as DataFrame
df_HackerNewsDataset = pd.read_csv(inputShpFile)
new_columns = df_HackerNewsDataset.columns.values.tolist()
new_columns[0] = 'index'
df_HackerNewsDataset.columns = new_columns
df_HackerNewsDataset.set_index("index" ,inplace=True)
print(df_HackerNewsDataset)
# Your code will parse the
# files in the training set and build a vocabulary with
# all the words it contains in Title which is Created At 2018.
df_trainingSet2018 = df_HackerNewsDataset[df_HackerNewsDataset.year == 2018]
print(df_trainingSet2018)
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



def parser(df_toParse):
    dict_vocabulary = {"allWordsInTitle" :[], "Post Type" :[] ,"Title" :[]}
    for row in df_toParse.iterrows():
        vocabulariesInRow = row[1].Title.split()
        rowPostType = row[1]["Post Type"]
        rowTitle = row[1]["Title"]
        for v in vocabulariesInRow:
            dict_vocabulary["allWordsInTitle"].append(v)
            dict_vocabulary["Post Type"].append(rowPostType)
            dict_vocabulary["Title"].append(rowTitle)
    return dict_vocabulary

print(df_trainingSet2018["Post Type"].unique())
lista = parser(df_trainingSet2018)
print(lista["Post Type"])