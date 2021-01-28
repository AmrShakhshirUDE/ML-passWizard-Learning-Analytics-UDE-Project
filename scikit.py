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
    #print the accuracy score of the model
    print("Naive-Bayes accuracy : ",accuracy_score(y_test, pred, normalize = True))


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

#encoding attributes
le = preprocessing.LabelEncoder()
#original features ~ 5
dataPro['school']       =le.fit_transform(dataPro['school'])
dataPro['failures']     =le.fit_transform(dataPro['failures'])
dataPro['higher']       =le.fit_transform(dataPro['higher'])
dataPro['Dalc']         =le.fit_transform(dataPro['Dalc'])
dataPro['absences']     =le.fit_transform(dataPro['absences'])
#merged/ Generated features ~ 3
dataPro['alcStudyTime'] =le.fit_transform(dataPro['alcStudyTime'])
dataPro['WalcAbsence']  =le.fit_transform(dataPro['WalcAbsence'])
dataPro['parentsEdu']   =le.fit_transform(dataPro['parentsEdu'])
#target values ~ 3
dataPro['GN1']          =le.fit_transform(dataPro['GN1'])
dataPro['GN2']          =le.fit_transform(dataPro['GN2'])
dataPro['GN3']          =le.fit_transform(dataPro['GN3'])
#derived target values ~ 3
dataPro['PF1']          =le.fit_transform(dataPro['PF1'])
dataPro['PF2']          =le.fit_transform(dataPro['PF2'])
dataPro['PF3']          =le.fit_transform(dataPro['PF3'])

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
    failures    = int(request.json['failures'])
    higher      = int(request.json['higher'])
    Dalc        = int(request.json['Dalc'])
    Walc        = int(request.json['Walc'])
    studytime   = int(request.json['studytime'])
    school      = int(request.json['school'])
    absences    = int(request.json['absences'])
    Fedu        = int(request.json['Fedu'])
    Medu        = int(request.json['Medu'])

    sTimeTemp   = (1-(studytime/3)) * 199
    alcSudyTime = (sTimeTemp + Dalc * 200)/399

    WalcTemp    = Walc * 199
    AbsenceTemp = absences * 200
    WalcAbsence = (WalcTemp + AbsenceTemp)/399

    parentsEdu = (3 * Medu + Fedu)/4
    WalcAbsence = (Walc * 199 + absences * 200)/399

    predFP1 = modelPF1.predict([[school,absences,higher,alcSudyTime,failures,WalcAbsence]])

    predFP2 = modelPF2.predict([[failures,higher,absences,WalcAbsence,parentsEdu]])

    predFP3 = modelPF3.predict([[Dalc,school,absences,higher,parentsEdu,failures,WalcAbsence]])

    #res= jsonify({'msg': 'Valuse have not been assigned properly!!})
    messageRes1=''
    messagePer1=''
    messageRes2=''
    messagePer2=''
    messageRes3=''
    messagePer3=''
    #add PF1 msg
    if predFP1 == 0:
        messagePer1 += ('86.36%')
        messageRes1 += ('fail')
    else:
        messagePer1 += ('88.89%')
        messageRes1 += ('pass')
    #add PF2 msg
    if predFP2 == 0:
        messagePer2 += ('91.67%')
        messageRes2 += ('fail')
    else:
        messagePer2 += ('80.51%')
        messageRes2 += ('pass')
    #add PF3 msg
    if predFP3 == 0:
        messagePer3 += ('55.56%')
        messageRes3 += ('fail')
    else:
        messagePer3 += ('85.12%')
        messageRes3 += ('pass')
    
    res= jsonify({'msg1': messagePer1,'msg2': messageRes1,
                    'msg3': messagePer2,'msg4': messageRes2,
                    'msg5': messagePer3,'msg6': messageRes3})
    return res


@app.route('/predict/por/G2',methods=['POST'])
def predictG2():
    failures    = int(request.json['failures'])
    higher      = int(request.json['higher'])
    Walc        = int(request.json['Walc'])
    absences    = int(request.json['absences'])
    Fedu        = int(request.json['Fedu'])
    Medu        = int(request.json['Medu'])
    G1          = int(request.json['G1'])

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
    
    messageResG2=''
    messagePerG2=''
    if predG2 == 0:
        messagePerG2 = ('78.125%')
        messageResG2 = ('fail')
    elif predG2 == 1:
        messagePerG2 = ('87.8%')
        messageResG2 = ('medium')
    else:
        messagePerG2 = ('81.25%')
        messageResG2 = ('high')
    
    res = jsonify({'msg33': messagePerG2,'msg44': messageResG2})
    return res

@app.route('/predict/por/G3',methods=['POST'])
def predictG3():
    G1          = int(request.json['G1'])
    G2          = int(request.json['G2'])

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
    print(predG3)
    messageResG3=''
    messagePerG3=''
    if predG3 == 0:
        messagePerG3 = ('66.67%')
        messageResG3 = ('fail')
    elif predG3 == 1:
        messagePerG3 = ('87.64%')
        messageResG3 = ('medium')
    else:
        messagePerG3 = ('94.11%')
        messageResG3 = ('high')
    
    res = jsonify({'msg55': messagePerG3,'msg66': messageResG3})    
    return res

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
