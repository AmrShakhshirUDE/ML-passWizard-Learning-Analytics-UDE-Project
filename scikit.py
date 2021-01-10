import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import graphviz #to export tree in decsion
#FeatureSelection using chi2
from sklearn.feature_selection import SelectKBest, chi2
#FeatureSelection using recursive
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
#FeatureSelection using SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#Naive-Bais
from sklearn.naive_bayes import GaussianNB
#LinearSVC "Support Vector Classifier"
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
#K-Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#Decision Tree
from sklearn import tree

#Vis. Comparision
from yellowbrick.classifier import ClassificationReport
#Evaluating
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

#statistical measures
from sklearn.feature_selection import SelectKBest, chi2


# # random seed
# np.random.seed(1)

# f=open('numericalValues.csv')
# f.readline()  # skip the header
# data = np.loadtxt(f)

# url="./finalComb.csv"
# url="./originalComb.csv"
# url="./modfirstCombined.csv"
url="./numValuesComb.csv"
# url="./newAttr.csv"
# url="./numInOut.csv"
# dataPro=pd.read_csv(url,error_bad_lines=False)
dataPro=pd.read_csv(url, sep=' ')

# # caricamento dati
# iris_dataset = datasets.load_iris()
# print (dataPro)
# print(dataPro.dtypes)

# #Plot
# # set the background colour of the plot to white
# sns.set(style="whitegrid", color_codes=True)
# # setting the plot size for all plots
# sns.set(rc={'figure.figsize':(11.7,8.27)})
# create a countplot

#barchart
# sns.countplot('school',data=dataPro,hue = ' G1')

#scatter
# sns.scatterplot(data=dataPro, x=' sex', y=' G1', hue='school', style_order=' G1')

#stacked bar
# sns.set()
# dataPro.set_index(' sex').T.plot(kind='bar', stacked=True)

# Remove the top and down margin
# sns.despine(offset=10, trim=True)
# display
# plt.show()




#Naive-Bais
# def naive_bayes(data, target):
def naive_bayes(X_train, X_test, y_train, y_test):
    # # separatione data into train & test
    # X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state = 10) #'random_state' just ensures that we get reproducible results every time.
    #create an object of the type GaussianNB
    gnb = GaussianNB()
    #train the algorithm on training data and predict using the testing data
    pred = gnb.fit(X_train, y_train.values.ravel()).predict(X_test)
    #print(pred.tolist())
    #print the accuracy score of the model
    print("Naive-Bayes accuracy : ",accuracy_score(y_test, pred, normalize = True))
    # #F1 scores
    # print('precision_recall_fscore_support: ',precision_recall_fscore_support(y_test, pred, average='macro'))
    # print(f1_score(y_test,pred, average=None))

# # Instantiate the classification model and visualizer
# # visualizer = ClassificationReport(gnb, classes=['fail','pass','good','V.good','excellent'])
# classes=['fail','pass']
# visualizer = ClassificationReport(y_test, pred, target)
# visualizer.fit(X_train, y_train) # Fit the training data to the visualizer
# visualizer.score(X_test, y_test) # Evaluate the model on the test data
# g = visualizer.poof() # Draw/show/poof the data

#Support Vector Classification LinearSVC
# def SVC(data, target):
def SVC(X_train, X_test, y_train, y_test):
    # # separatione data into train & test
    # X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state = 10) #'random_state' just ensures that we get reproducible results every time.
    #create an object of type LinearSVC
    svc_model = LinearSVC(random_state=0) #instruction to the built-in random number generator to shuffle the data in a specific order
    #train the algorithm on training data and predict using the testing data
    pred = svc_model.fit(X_train, y_train.values.ravel()).predict(X_test)
    #print the accuracy score of the model
    print("LinearSVC accuracy : ",accuracy_score(y_test, pred, normalize = True))

# # Instantiate the classification model and visualizer
# visualizer = ClassificationReport(pred, classes=['fail','pass','good','V.good','excellent'])
# # visualizer.fit(X_train, y_train) # Fit the training data to the visualizer
# # visualizer.score(X_test, y_test) # Evaluate the model on the test data
# # g = visualizer.poof() # Draw/show/poof the data

#K-Neighbors Classifier
# def kNeigh(data, target):
def kNeigh(X_train,X_test,y_train, y_test):
    # # separatione data into train & test
    # X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state = 10) #'random_state' just ensures that we get reproducible results every time.
    #create object of the classifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    #Train the algorithm
    neigh.fit(X_train, y_train.values.ravel())
    # predict the response
    pred = neigh.predict(X_test)
    # evaluate accuracy
    print ("KNeighbors accuracy score : ",accuracy_score(y_test, pred))

# # Instantiate the classification model and visualizer
# visualizer = ClassificationReport(pred, classes=['fail','pass','good','V.good','excellent'])
# visualizer.fit(X_train, y_train) # Fit the training data to the visualizer
# visualizer.score(X_test, y_test) # Evaluate the model on the test data
# g = visualizer.poof() # Draw/show/poof the data

# Decision Tree
# def decTree(data, target):
def decTree(X_train,X_test,y_train, y_test):
    # # separatione data into train & test
    # X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state = 10) #'random_state' just ensures that we get reproducible results every time.
    #create object of the classifier
    dTree = tree.DecisionTreeClassifier()
    #Train the algorithm
    # X_train = SelectKBest(chi2, k=2).fit_transform(X_train, y_train.values.ravel())
    dTree = dTree.fit(X_train, y_train.values.ravel())
    # predict the response
    pred = dTree.predict(X_test)
    # evaluate accuracy
    print ("Decision Tree accuracy score : ",accuracy_score(y_test, pred))

#encoding attributes
le = preprocessing.LabelEncoder()
dataPro['school']=le.fit_transform(dataPro['school'])
dataPro['sex']=le.fit_transform(dataPro['sex'])
dataPro['age']=le.fit_transform(dataPro['age'])
dataPro['address']=le.fit_transform(dataPro['address'])
dataPro['famsize']=le.fit_transform(dataPro['famsize'])
dataPro['Pstatus']=le.fit_transform(dataPro['Pstatus'])
dataPro['Medu']=le.fit_transform(dataPro['Medu'])
dataPro['Fedu']=le.fit_transform(dataPro['Fedu'])
dataPro['Mjob']=le.fit_transform(dataPro['Mjob'])
dataPro['Fjob']=le.fit_transform(dataPro['Fjob'])
dataPro['reason']=le.fit_transform(dataPro['reason'])
dataPro['guardian']=le.fit_transform(dataPro['guardian'])
dataPro['traveltime']=le.fit_transform(dataPro['traveltime'])
dataPro['studytime']=le.fit_transform(dataPro['studytime'])
dataPro['failures']=le.fit_transform(dataPro['failures'])
dataPro['schoolsup']=le.fit_transform(dataPro['schoolsup'])
dataPro['famsup']=le.fit_transform(dataPro['famsup'])
dataPro['paid']=le.fit_transform(dataPro['paid'])
dataPro['activities']=le.fit_transform(dataPro['activities'])
dataPro['nursery']=le.fit_transform(dataPro['nursery'])
dataPro['higher']=le.fit_transform(dataPro['higher'])
dataPro['internet']=le.fit_transform(dataPro['internet'])
dataPro['romantic']=le.fit_transform(dataPro['romantic'])
dataPro['famrel']=le.fit_transform(dataPro['famrel'])
dataPro['freetime']=le.fit_transform(dataPro['freetime'])
dataPro['goout']=le.fit_transform(dataPro['goout'])
dataPro['Dalc']=le.fit_transform(dataPro['Dalc'])
dataPro['Walc']=le.fit_transform(dataPro['Walc'])
dataPro['health']=le.fit_transform(dataPro['health'])
dataPro['absences']=le.fit_transform(dataPro['absences'])
dataPro['FailHigh']=le.fit_transform(dataPro['FailHigh'])
dataPro['alcSudyTime']=le.fit_transform(dataPro['alcSudyTime'])
dataPro['parentsEdu']=le.fit_transform(dataPro['parentsEdu'])
dataPro['GC1']=le.fit_transform(dataPro['GN1'])
dataPro['GC2']=le.fit_transform(dataPro['GN2'])
dataPro['GC3']=le.fit_transform(dataPro['GN3'])


cols=["school","sex","age","address",'famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason',
'guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities',
'nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences',
'FailHigh','alcSudyTime','parentsEdu']

# cols=["school","sex",'Pstatus','studytime','failures','schoolsup','paid','activities',
# 'higher','famrel','Dalc',] #selectFromModel SVC 70.76% l2

# cols=['studytime','failures','famrel','Dalc',] #selectFromModel SVC 66.96% l2 then l1

# cols=["age",'Medu','Fedu','Mjob','Fjob','reason','studytime','failures','famrel','freetime','absences',
# 'FailHigh','alcSudyTime','parentsEdu'] #selectFromModel SVC 63.84% l1

# cols=['Medu','reason','studytime','failures','famrel'] #selectFromModel SVC 66.92% l1 then l2

# cols=["school","age",'Medu','studytime','failures','higher','Dalc','absences','alcSudyTime','parentsEdu'] #Choosen by recursive & chi2 68.46% SVL


# cols=["age",'Medu','Fedu','Mjob','Fjob','studytime','failures','higher',
# 'famrel','absences','FailHigh','alcSudyTime','parentsEdu'] #Choosen by recursive 66.15% SVL

# cols=['failures','absences','alcSudyTime'] #Choosen by recursive & chi2 66.92% SVL

# cols=['failures','higher','absences','Dalc',"school"] #Choosen by chi2 70.76% SVL
# cols=['school','Medu','Fedu','studytime','Dalc','failures','higher']
# cols=['school','Medu','studytime','Dalc','failures','higher']
# cols=['school','Medu','Fedu','studytime','failures','higher']
# cols=['school','Fedu','studytime','failures','higher'] # same as above for GC1 but higher accuarcy in Kmeans & decisionTree
# cols=['school','Fedu','studytime','failures','higher'] # same as above for GC1 but higher accuarcy in Kmeans & decisionTree

# cols=['GC1','GC2']  
# cols=['GC3']
# cols=['goout','studytime','Mjob','parentsEdu','FailHigh','Medu']# highest for Naive-Bais 63.84
# cols=['higher','failures','school','age','absences']# highest for SVC 72.30
# cols=['age','guardian','failures','higher']# highest for dTree 71.53




data = dataPro[cols]
#assigning the Oppurtunity Result column as target
# tarCol = [' GC1',' GC2',' GC3']
tarCol= ['GN1']
target = dataPro[tarCol]

# separatione data into train & test
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state = 10) #'random_state' just ensures that we get reproducible results every time.

# # chi2 method of selecting features
# sel = SelectKBest(chi2,1)
# sel.fit(X_train,y_train)
# print(sel.get_support())
# # print(sel.scores_)


# #Recursive method for feature selection
# recSel= RFECV(RandomForestClassifier(),scoring='accuracy')
# recSel.fit(X_train,y_train.values.ravel())
# print(recSel.get_support())
# # print(recSel.fit(X_train,y_train.values.ravel()))
# # # print(recSel.score(X_train,y_train))

# SelectFromModel method for feature selection
sfm= SelectFromModel(LinearSVC(C=0.01, penalty='l2',dual=False))
sfm.fit(X_train,y_train.values.ravel())
print(sfm.get_support())
# # print(recSel.fit(X_train,y_train.values.ravel()))
# # # print(recSel.score(X_train,y_train))



# #get accuracy for each attributes individually
# for x in cols:
#     print(x)
#     col=[x]
#     data = dataPro[col]
#     decTree(data, target)
#     kNeigh(data, target)

# # naive_bayes(data, target)
# # SVC(data, target)
# # kNeigh(data, target)
# # decTree(data, target)
naive_bayes(X_train,X_test,y_train, y_test)
SVC(X_train,X_test,y_train, y_test)
kNeigh(X_train,X_test,y_train, y_test)
decTree(X_train,X_test,y_train, y_test)

#F1 scores
# print('precision_recall_fscore_support: ',precision_recall_fscore_support(y_test, pred, average='macro'))
# print(f1_score(y_test,pred, average=None))
# print(target.head(15))