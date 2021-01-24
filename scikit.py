#flask
from flask import Flask, jsonify, request, json,send_file,redirect,url_for,session
from flask_pymongo import PyMongo
from bson.json_util import dumps
from flask_cors import CORS
from pymongo import MongoClient

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
#from yellowbrick.classifier import ClassificationReport
#Evaluating
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score

#statistical measures
from sklearn.feature_selection import SelectKBest, chi2

###flask configuration
app = Flask(__name__)
CORS(app)
app.config["MONGO_URI"] = "mongodb://localhost:27017/ladb"
mongo = PyMongo(app)

###database configuration
server = "127.0.0.1"
port = 27017
db = "ladb"
collection = "passwizard" 

def _connect_mongo(host, port, username, password, db):
    """ A util for making a connection to mongo """

    if username and password:
        mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)


    return conn[db]


def read_mongo(db, collection, query={}, host='localhost', port=27017, username=None, password=None, no_id=True):
    """ Read from Mongo and Store into DataFrame """

    # Connect to MongoDB
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    # Make a query to the specific DB and Collection
    cursor = db[collection].find(query)

    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))

    # Delete the _id
    if no_id:
        del df['_id']

    return df
#create data object to use in machine learinig part
dataPro=read_mongo(db, collection)

###machine learning segment
#Naive-Bais
def naive_bayes(X_train, X_test, y_train, y_test):
    #create an object of the type GaussianNB
    gnb = GaussianNB()
    #train the algorithm on training data and predict using the testing data
    pred = gnb.fit(X_train, y_train.values.ravel()).predict(X_test)
    #print(pred.tolist())
    #print the accuracy score of the model
    print("Naive-Bayes accuracy : ",accuracy_score(y_test, pred, normalize = True))
    # #Precision Score
    # print('precision_support: ',precision_score(y_test, pred, average='micro'))
    # #F1 scores
    # print('precision_recall_fscore_support: ',precision_recall_fscore_support(y_test, pred, average='macro'))
    # print(f1_score(y_test,pred, average=None))

#Support Vector Classification LinearSVC
def SVC(X_train, X_test, y_train, y_test):
    #create an object of type LinearSVC
    svc_model = LinearSVC(random_state=0,dual=False) #instruction to the built-in random number generator to shuffle the data in a specific order
    #train the algorithm on training data and predict using the testing data
    pred = svc_model.fit(X_train, y_train.values.ravel()).predict(X_test)
    #print the accuracy score of the model
    print("LinearSVC accuracy : ",accuracy_score(y_test, pred, normalize = True))
    #Precision Score
    print('precision_support: ',precision_score(y_test, pred, average=None))
    # #F1 scores
    # print(f1_score(y_test,pred, average=None))

#K-Neighbors Classifier
def kNeigh(X_train,X_test,y_train, y_test):
    #create object of the classifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    #Train the algorithm
    neigh.fit(X_train, y_train.values.ravel())
    # predict the response
    pred = neigh.predict(X_test)
    # evaluate accuracy
    print ("KNeighbors accuracy score : ",accuracy_score(y_test, pred))

# Decision Tree
def decTree(X_train,X_test,y_train, y_test):
    #create object of the classifier
    dTree = tree.DecisionTreeClassifier()
    #Train the algorithm
    dTree = dTree.fit(X_train, y_train.values.ravel())
    # predict the response
    pred = dTree.predict(X_test)
    # evaluate accuracy
    print ("Decision Tree accuracy score : ",accuracy_score(y_test, pred))
    #Precision Score
    print('precision_support: ',precision_score(y_test, pred, average=None), dTree.classes_)
    # print(target.head(18))
    # print(target.tail(13))

#encoding attributes
le = preprocessing.LabelEncoder()
dataPro['school']       =le.fit_transform(dataPro['school'])
dataPro['failures']     =le.fit_transform(dataPro['failures'])
dataPro['higher']       =le.fit_transform(dataPro['higher'])
dataPro['Dalc']         =le.fit_transform(dataPro['Dalc'])
dataPro['absences']     =le.fit_transform(dataPro['absences'])
dataPro['alcStudyTime'] =le.fit_transform(dataPro['alcStudyTime'])
dataPro['WalcAbsence']  =le.fit_transform(dataPro['WalcAbsence'])
dataPro['parentsEdu']   =le.fit_transform(dataPro['parentsEdu'])
dataPro['GN1']          =le.fit_transform(dataPro['GN1'])
dataPro['GN2']          =le.fit_transform(dataPro['GN2'])
dataPro['GN3']          =le.fit_transform(dataPro['GN3'])
dataPro['PF1']          =le.fit_transform(dataPro['PF1'])
dataPro['PF2']          =le.fit_transform(dataPro['PF2'])
dataPro['PF3']          =le.fit_transform(dataPro['PF3'])
# dataPro['sex']          =le.fit_transform(dataPro['sex'])
# dataPro['age']          =le.fit_transform(dataPro['age'])
# dataPro['address']      =le.fit_transform(dataPro['address'])
# dataPro['famsize']      =le.fit_transform(dataPro['famsize'])
# dataPro['Pstatus']      =le.fit_transform(dataPro['Pstatus'])
# dataPro['Medu']         =le.fit_transform(dataPro['Medu'])
# dataPro['Fedu']         =le.fit_transform(dataPro['Fedu'])
# dataPro['Mjob']         =le.fit_transform(dataPro['Mjob'])
# dataPro['Fjob']         =le.fit_transform(dataPro['Fjob'])
# dataPro['reason']       =le.fit_transform(dataPro['reason'])
# dataPro['guardian']     =le.fit_transform(dataPro['guardian'])
# dataPro['traveltime']   =le.fit_transform(dataPro['traveltime'])
# dataPro['studytime']    =le.fit_transform(dataPro['studytime'])
# dataPro['schoolsup']    =le.fit_transform(dataPro['schoolsup'])
# dataPro['famsup']       =le.fit_transform(dataPro['famsup'])
# dataPro['paid']         =le.fit_transform(dataPro['paid'])
# dataPro['activities']   =le.fit_transform(dataPro['activities'])
# dataPro['nursery']      =le.fit_transform(dataPro['nursery'])
# dataPro['internet']     =le.fit_transform(dataPro['internet'])
# dataPro['romantic']     =le.fit_transform(dataPro['romantic'])
# dataPro['famrel']       =le.fit_transform(dataPro['famrel'])
# dataPro['freetime']     =le.fit_transform(dataPro['freetime'])
# dataPro['goout']        =le.fit_transform(dataPro['goout'])
# dataPro['Walc']         =le.fit_transform(dataPro['Walc'])
# dataPro['health']       =le.fit_transform(dataPro['health'])
# dataPro['FailHigh']     =le.fit_transform(dataPro['FailHigh'])

# cols=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason',
# 'guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities',
# 'nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences',
# 'studytime','WalcAbsence','parentsEdu','GN1','GN2']


################################## 3 CATEGORIES [FAIL-MODERATE-HIGH] RATIO 80/20 ################################################

colsPF1=['school','absences','higher','alcStudyTime','failures','WalcAbsence']#PF1~ ch2 2 categories 0.8846 [0.8636 0.8888]
# colsG1=['school','absences','higher','alcStudyTime','failures','WalcAbsence']#G1~ ch2 3 categories 0.7538 [0.8260 0.7383 0.0000]

colsPF2=['failures','higher','absences','WalcAbsence','parentsEdu']#PF2~ ch2 2 categories 0.8153 [0.9166 0.8050]
# cols=['school','failures','higher','absences','WalcAbsence','parentsEdu']#G2~ ch2 3 categories 0.7384 [0.8888 0.7142 0.0000]
colsG2=['failures','higher','absences','WalcAbsence','parentsEdu','PF1','GN1']#G2~ ch2 3 categories 0.84615 [0.78125 0.8780 0.8125]

colsPF3=['Dalc','school','absences','higher','parentsEdu','failures','WalcAbsence']#PF3~ ch2 2 categories 0.8307 [0.5555 0.8512]
# cols=['GN1','GN2']#PF3~ 2 categories 0.8846 [0.6666 0.9339]
# cols=['Dalc','school','absences','higher','parentsEdu','failures','WalcAbsence']#G3~ ch2 3 categories 0.6923 [0.6363 0.6974 0.0000]
colsG3=['PF1','PF2','GN1','GN2']#G3~ ch2 3 categories 0.8461 [0.6666 0.8764 0.9411] 

################################################### END OF 3 CATEGORIES #################################################################

dataPF1 = dataPro[colsPF1]
dataPF2 = dataPro[colsPF2]
dataG2  = dataPro[colsG2]
dataPF3 = dataPro[colsPF3]
dataG3  = dataPro[colsG3]

targetPF1   = dataPro['PF1']
targetPF2   = dataPro['PF2']
targetG2    = dataPro['GN2']
targetPF3   = dataPro['PF3']
targetG3    = dataPro['GN3']

##########################################################PF1 model######################################################################
X_train, X_test, y_train, y_test = train_test_split(dataPF1, targetPF1, test_size=0.2, random_state = 10) #'random_state' just ensures that we get reproducible results every time.
#create an object of type LinearSVC
svc_model = LinearSVC(random_state=0,dual=False) #instruction to the built-in random number generator to shuffle the data in a specific order
#train the algorithm on training data and predict using the testing data
# global modelPF1 
modelPF1 = svc_model.fit(X_train, y_train.values.ravel())
predFP1 = modelPF1.predict(X_test)
print ("SVC PF1 accuracy score : ",accuracy_score(y_test, predFP1))
#Precision Score
print('precision_support: ',precision_score(y_test, predFP1, average=None))
##########################################################End PF1 model##################################################################



##########################################################PF2 model######################################################################
X_train, X_test, y_train, y_test = train_test_split(dataPF2, targetPF2, test_size=0.2, random_state = 10) #'random_state' just ensures that we get reproducible results every time.
#create an object of type LinearSVC
svc_model = LinearSVC(random_state=0,dual=False) #instruction to the built-in random number generator to shuffle the data in a specific order
#train the algorithm on training data and predict using the testing data
modelPF2 = svc_model.fit(X_train, y_train.values.ravel())
predFP2 = modelPF2.predict(X_test)
print ("SVC PF2 accuracy score : ",accuracy_score(y_test, predFP2))
#Precision Score
print('precision_support: ',precision_score(y_test, predFP2, average=None))
##########################################################End PF2 model##################################################################


##########################################################G2 model######################################################################
X_train, X_test, y_train, y_test = train_test_split(dataG2, targetG2, test_size=0.2, random_state = 10) #'random_state' just ensures that we get reproducible results every time.
#create an object of type LinearSVC
svc_model = LinearSVC(random_state=0,dual=False) #instruction to the built-in random number generator to shuffle the data in a specific order
#train the algorithm on training data and predict using the testing data
modelG2 = svc_model.fit(X_train, y_train.values.ravel())
predG2 = modelG2.predict(X_test)
print ("SVC G2 accuracy score : ",accuracy_score(y_test, predG2))
#Precision Score
print('precision_support: ',precision_score(y_test, predG2, average=None))
##########################################################End G2 model##################################################################


##########################################################PF3 model######################################################################
X_train, X_test, y_train, y_test = train_test_split(dataPF3, targetPF3, test_size=0.2, random_state = 10) #'random_state' just ensures that we get reproducible results every time.
#create an object of type LinearSVC
svc_model = LinearSVC(random_state=0,dual=False) #instruction to the built-in random number generator to shuffle the data in a specific order
#train the algorithm on training data and predict using the testing data
modelPF3 = svc_model.fit(X_train, y_train.values.ravel())
predFP3 = modelPF3.predict(X_test)
print ("SVC PF3 accuracy score : ",accuracy_score(y_test, predFP3))
#Precision Score
print('precision_support: ',precision_score(y_test, predFP3, average=None))
##########################################################End PF3 model##################################################################

##########################################################G3 model######################################################################
X_train, X_test, y_train, y_test = train_test_split(dataG3, targetG3, test_size=0.2, random_state = 10) #'random_state' just ensures that we get reproducible results every time.
#create an object of type LinearSVC
svc_model = LinearSVC(random_state=0,dual=False) #instruction to the built-in random number generator to shuffle the data in a specific order
#train the algorithm on training data and predict using the testing data
modelG3 = svc_model.fit(X_train, y_train.values.ravel())
predG3 = modelG3.predict(X_test)
print ("SVC G3 accuracy score : ",accuracy_score(y_test, predG3))
#Precision Score
print('precision_support: ',precision_score(y_test, predG3, average=None))
##########################################################End G3 model##################################################################

@app.route("/alldata")
def home_page():
    df = mongo.db.passwizardfe.find()
    resp = dumps(df)
    return resp

@app.route('/predict/por/pf',methods=['POST'])
def predictPF1():
    failures    = int(request.form['failures'])
    higher      = int(request.form['higher'])
    Dalc        = int(request.form['Dalc'])
    Walc        = int(request.form['Walc'])
    studytime   = int(request.form['studytime'])
    school      = int(request.form['school'])
    absences    = int(request.form['absences'])
    Fedu        = int(request.form['Fedu'])
    Medu        = int(request.form['Medu'])

    sTimeTemp   = (1-(studytime/3)) * 199
    alcSudyTime = (sTimeTemp + Dalc * 200)/399

    WalcTemp    = Walc * 199
    AbsenceTemp = absences * 200
    WalcAbsence = (WalcTemp + AbsenceTemp)/399

    parentsEdu = (3 * Medu + Fedu)/4
    WalcAbsence = (Walc * 199 + absences * 200)/399

    predFP1 = modelPF1.predict([[school,absences,higher,alcSudyTime,failures,WalcAbsence]])
    # predFP1 = modelPF1.predict([[0,1,1,0.5,5,0]])
    # predFP1 = modelPF1.predict([[0,0,0,0.5,0,0]])
    # print('predFP1')
    # print(predFP1)

    predFP2 = modelPF2.predict([[failures,higher,absences,WalcAbsence,parentsEdu]])
    # predFP2 = modelPF2.predict([[0,1,1,0.5,5]])
    # print('predFP2')
    # print(predFP2)

    predFP3 = modelPF3.predict([[Dalc,school,absences,higher,parentsEdu,failures,WalcAbsence]])
    # predFP3 = modelPF3.predict([[0,1,1,0.5,5,0,0]])
    # print('predFP3')
    # print(predFP3)

    #res= jsonify({'msg': 'Valuse have not been assigned properly!!})
    messageRes=''
    #add PF1 msg
    if predFP1 == 0:
        messageRes += ('there is a chance of 86.36% to fail at the first exam!\n ')
    else:
        messageRes += ('there is a chance of 88.89% to pass at the first exam!\n ')
    #add PF2 msg
    if predFP2 == 0:
        messageRes += ('there is a chance of 91.67% to fail at the second exam!\n ')
    else:
        messageRes += ('there is a chance of 80.51% to pass at the second exam!\n ')
    #add PF3 msg
    if predFP3 == 0:
        messageRes += ('there is a chance of 55.56% to fail at the final exam!')
    else:
        messageRes += ('there is a chance of 85.12% to pass at the final exam!')
    
    res= jsonify({'msg': messageRes})
    return res
    
    return res

    # return'''
    # <html>HelloWorld PF</html'''


@app.route('/predict/por/G2',methods=['POST'])
def predictG2():
    failures    = int(request.form['failures'])
    higher      = int(request.form['higher'])
    Walc        = int(request.form['Walc'])
    absences    = int(request.form['absences'])
    Fedu        = int(request.form['Fedu'])
    Medu        = int(request.form['Medu'])
    G1          = int(request.form['G1'])

    parentsEdu = (3 * Medu + Fedu)/4
    WalcAbsence = (Walc * 199 + absences * 200)/399

    if G1 < 10:
        PF1 = 0
    else:
        PF1 = 1
    
    if G1 < 10:
        GN1 = 0
    elif G1>= 10 and G1 < 15:
        GN1 = 1
    else:
        GN1 = 2

    predG2 = modelG2.predict([[failures,higher,absences,WalcAbsence,parentsEdu,PF1,GN1]])
    # predG2 = modelG2.predict([[0,1,1,0.5,5,0,0]])
    # print('predG2')
    # print(predG2)

    if predG2 == 0:
        res= jsonify({'msg':'there is a chance of 78.125% to fail at the second exam!'})
    elif predG2 == 1:
        res= jsonify({'msg':'there is a chance of 87.8% to pass with a medium grade at the second exam!'})
    else:
        res= jsonify({'msg':'there is a chance of 81.25% to pass with a high grade at the second exam!'})
    
    return res


    # return'''
    # <html>HelloWorld G2</html'''

@app.route('/predict/por/G3',methods=['POST'])
def predictG3():
    G1          = int(request.form['G1'])
    G2          = int(request.form['G2'])

    if G1 < 10:
        PF1 = 0
    else:
        PF1 = 1
    
    if G1 < 10:
        GN1 = 0
    elif G1>=10 and G1 < 15:
        GN1 = 1
    else:
        GN1 = 2

    if G2 < 10:
        PF2 = 0
    else:
        PF2 = 1
    
    if G2 < 10:
        GN2 = 0
    elif G2>=10 and G2 < 15:
        GN2 = 1
    else:
        GN2 = 2

    predG3 = modelG3.predict([[PF1,PF2,GN1,GN2]])
    # predG3 = modelG3.predict([[0,1,1,1]])
    # print('predG3')
    print(predG3)

    if predG3 == 0:
        res= jsonify({'msg':'there is a chance of 66.67% to fail at the final exam!'})
    elif predG3 == 1:
        res= jsonify({'msg':'there is a chance of 87.64% to pass with a medium grade at the final exam!'})
    else:
        res= jsonify({'msg':'there is a chance of 94.11% to pass with a high grade at the final exam!'})
    
    return res

    # return'''
    # <html>HelloWorld G3</html'''

@app.route('/')  #check connectivity
def connected():
    return'''
    <html>
        <h1>Connected to LAPro-Group Project Backend Side!!!</h1>
        <h4>Team members in alphabetical order</h4>
        <ul>
        <li>Amr Shakhshir</li>
        <li>Baohui Deng</li>
        <li>Hesamoddin Heidarzadeh</li>
        <li>Tannaz Vahidi</li>
        </ul>
    </html>
    '''


if __name__ == '__main__':
    app.run(debug =True )