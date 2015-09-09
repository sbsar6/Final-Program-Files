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
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
            
def machLearn(path):
    
    df = pd.read_csv(path)
    #DataCSV['Playing_Location'] = DataCSV['Playing_Location'].apply(lambda x: str(x).replace("home","1").replace("away","0"))
    #DataCSV['Penalties_Conceeded'] = DataCSV['Penalties_Conceeded'].apply(lambda x: float(str(x).split("(")[0])) 
    
    List = ["Unnamed:_0","LineBreakPrevPhase", "PhaseName", "Tries","PhaseID"]
    columns = df.columns.values
    Features = [i for i in columns if str(i) not in List]
    bucket = df["Tries"].tolist()
    freqs = Counter(bucket)
    print(freqs)
    X =  np.array(df[Features].values)
    X = preprocessing.scale(X,axis=0)
        
    y = np.array(df["Tries"].values)
    
    
    
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
    
    bar = Bar(ranks,features,title="Dragons Feature Rankings Phase Tries",palette=brewer["Reds"][4], stacked=True)
    
    output_file("PhaseTrankings.html")
    show(bar)

    '''
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y, 2),
                  scoring='accuracy')
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    ''' 
    

path = "all_gamesPhase.csv"
machLearn(path)


