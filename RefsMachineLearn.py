import pandas as pd
from sklearn import linear_model, svm, preprocessing
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn import cross_validation
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from bokeh.palettes import brewer
palette = brewer["Blues"][3]
from bokeh.plotting import *
from bokeh.charts import Bar, show, output_file
import bokeh
from bokeh.plotting import figure, show
#from bokeh.palettes import Green3
import json

            
def machLearn(path):
    
    DataCSV = pd.read_csv(path)
    #DataCSV['Playing_Location'] = DataCSV['Playing_Location'].apply(lambda x: str(x).replace("home","1").replace("away","0"))
    #DataCSV['Penalties_Conceeded'] = DataCSV['Penalties_Conceeded'].apply(lambda x: float(str(x).split("(")[0])) 
    
    List = ["Unnamed:_0","Team_Name","Referee","Result"]
    columns = DataCSV.columns.values
    Features = [i for i in columns if str(i) not in List]
    
    Labels = [1.0,0.0]
    
    #Populating a dictionary accordingly with each label
    dictionary = {}
    for Label in Labels:
        dictionary[Label] = DataCSV[DataCSV["Referee"] == Label]
    
    bucket = DataCSV["Referee"].tolist()
    freqs = Counter(bucket)
    print(freqs)
    
    #Features dictionary
    FeaturesDictionary = {}
    for Label in Labels:
        DataFeatures = dictionary[Label]
        
        #Creating a list with unique values, it works as an identifier for each row 
        UniqueValuesList = DataFeatures.index.tolist()
        features_list = []
        for j in range(len(UniqueValuesList)):
            Data = []
            for each_feature in range(len(Features)):
                value = DataFeatures[DataFeatures.index==UniqueValuesList[j]][Features[each_feature]].tolist()[0]
                Data.append(value)
            features_list.append(Data)
        FeaturesDictionary[Label] = features_list
        
    ValuesFeatures = [FeaturesDictionary[key] for key in FeaturesDictionary.keys()]
    X = ValuesFeatures[1] + ValuesFeatures[0]
    X = preprocessing.scale(X,axis=0)
    ones_list = [1] * freqs[1] 
    zeroes_list = [0] * freqs[0]
    
    y = np.array(ones_list + zeroes_list)
    
    
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y)
    
    clf = svm.SVC(kernel="linear")
    clf.fit(X_train,y_train)
    rf = RandomForestClassifier(n_estimators = 150)
    rf.fit(X_train,y_train)
    
    
    scores_SVM = clf.score(X_test,y_test)
    scores_RF = rf.score(X_test,y_test)
    rfe = RFE(svm.SVC(kernel="linear"), n_features_to_select=1)
    rfe.fit(X,y)
        
    ranks = rfe.ranking_
    
    clf = ExtraTreesClassifier(n_estimators = 250,random_state=0)
    clf.fit(X,y)
    importance = clf.feature_importances_
    
    sorted_index = np.argsort(importance)[::-1]
    
    print("acurracy SVM:",scores_SVM)
    print("acurracy Random Forest:",scores_RF)

    place = 1
    
    rankings = {}
    
    print("results using TreesClassifier")
    for i,j in zip(np.array(Features)[sorted_index],importance[sorted_index]):
        rankings[place] = [i,j]
        print("rank: {0}, {1}, {2}".format(place,i,j))
        place += 1
    
    print("----------------------------")
    print("----------------------------")
    place = 1
    print("results using RFE")
    print(sorted(zip(map(lambda x: round(x, 4), ranks), Features)))   
    
    ranks = [value[1] for value in rankings.values()]
    features = [value[0] for value in rankings.values()]
    
    bar = Bar(ranks,features,title="2 Referees Prediction",palette=brewer["Purples"][3], stacked=True)
    
    output_file("rankings.html")
    show(bar)

    
    return ranks, importance, rankings, Features  


path = "EP7SeasonsCleaned.csv"
machLearn(path)


