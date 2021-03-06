# Pass Wizard
This project is designed to predict students' grades in two secodary schools [Gabriel Pereira, Mousinho da Silveira] in Portugal in Portuguese Language Course.
> Project for the course: Learning Analytics, International studies in engineering, Master of computer engineering, University of Duisburg-Essen.

# Features
## Users can do the following:
1. Check how attributes and features are correlated to exams' grades.
2. Have a clear idea about their predicted grades through an interactive decision trees.
3. Get the predicted result (fail/ pass) for all exams (first, second, and final) at the beginning of the semester.
4. Get the predicted result (fail, medium, and high) for second exam after getting first exam's result<br />(fail: less than 10, medium: from 10 to less than 15, high: from 15 to 20).
5. Get the predicted result (fail, medium, and high) for final exam after getting second exam's result<br />(fail: less than 10, medium: from 10 to less than 15, high: from 15 to 20).

# Try on Heroku
[Pass Wizard](https://passwizard.herokuapp.com/)

# Live Demo
[Live demo and screenshots](https://www.youtube.com/watch?v=M09m9P4qvKg)

# Advertisement
[Advertisement Video](https://biteable.com/watch/educational-copy-2652764)

# Screenshots
>Logo

![logo](https://github.com/AmrShakhshirUDE/MLpassWizard/blob/main/Images/1.Logo.png)

>Landing page

![LandingPage](https://github.com/AmrShakhshirUDE/MLpassWizard/blob/main/Images/2.LandingPage.png)

>Footer

![Footer](https://github.com/AmrShakhshirUDE/MLpassWizard/blob/main/Images/3.Footer.png)

>Dataset

![Dataset](https://github.com/AmrShakhshirUDE/MLpassWizard/blob/main/Images/4.Datasest.png)

>Interactive Decision Tree

![Interactive Decision Tree](https://github.com/AmrShakhshirUDE/MLpassWizard/blob/main/Images/5.Decisiontree.png)

>Prediction page

![PredictionPage](https://github.com/AmrShakhshirUDE/MLpassWizard/blob/main/Images/6.PredictionPage.png)

>Prediction Form

![Prediction Form](https://github.com/AmrShakhshirUDE/MLpassWizard/blob/main/Images/7.Predictionform.png)

>Prediction Results

![Prediction Result 1](https://github.com/AmrShakhshirUDE/MLpassWizard/blob/main/Images/8.PredictionResult1.png)

![Prediction Result 2](https://github.com/AmrShakhshirUDE/MLpassWizard/blob/main/Images/9.PredictionResult2.png)

![Prediction Result 3](https://github.com/AmrShakhshirUDE/MLpassWizard/blob/main/Images/10.PredictionResult3.png)


# Technical architecture
The application consists of two main parts:
* Backend: responsible for: <br />Machine Learning <br />Server-side web application logic, consists of a server, an application, and a database.
* Frontend: the part that users interact with it.

# Technologies/ libraries used
![Technologies](https://github.com/AmrShakhshirUDE/MLpassWizard/blob/main/Images/11.Technologies.png)
## Backend technologies
* Flask
* MongoDB for database
* Scikit-learn for machine learning
## Frontend technologies
* React
* Bootstrap
## Connecting frontend to backend
* Axios
* Postman: to test connectivity especially for 'POST' method
## Deploy technologies
* Gunicorn as a web server gateway interface "WSGI"
* mLab as a cloud database service
* Github
* Heroku

# How to run the project
> Make sure that you have mongoDB installed on your PC and we highly recommend you to use visual studio code as a code editor

## To run locally
> Make sure to have the file `numValuesComb.csv` in the same folder that contains `scikit.py`

1. Import `allData.csv` to MongoDB. [This](https://medium.com/analytics-vidhya/import-csv-file-into-mongodb-9b9b86582f34) tutorial will help you.

2. In the top level directory, go to `scikit.py` file

comment lines [67 to 98]

`def _connect_mongo(host, port, username, password, db):`
until
`dataPro=read_mongo(db, collection)`

then uncomment lines [100 and 101]

```
url="./numValuesComb.csv"
dataPro=pd.read_csv(url, sep=' ')
```
3. In flask configuration `app.config["MONGO_URI"] = "mongodb://localhost:27017/ladb"` <br />
change the the database name (lab) to the name you chose for your database in step 1

4. In database configuration change the following <br /> `db = "YOUR OWN DB NAME"' and 'collection = "YOUR COLLECTION'S NAME"`
5. In the route `@app.route("/alldata")` change `df = mongo.db.YOUR COLLECTION'S NAME.find()`
6. Open terminal and go to the path of `scikit.py` then type the following:
```
pip install -r requirements.txt
python scikit.py
```

first command will install all required python packages to run this app
second command will run the backend part of the app

The backend part should be running

7. Go to `client\src\contexts\urlContext.js`

comment line.3

`export const UrlContext = createContext('https://passwizardbackend.herokuapp.com/');`

uncomment line.4

`export const UrlContext = createContext('http://127.0.0.1:5000/');`

```
npm install
npm start
```

> If `npm start` doesn't work, do the following:
```
npm install rimraf -g
rimraf node_modules
npm start -- --reset-cache
```
then repeat step number 7

## To deploy
1. In the top level directory, go to `scikit.py` file

comment line.16

`app.config["MONGO_URI"] = "mongodb://localhost:27017/final"`

then uncomment line.15

`app.config["MONGO_URI"] = os.environ.get("MONGODB_URI")`

2. Go to `client\src\contexts\urlContext.js`

uncomment line.3

`export const UrlContext = createContext("https://first-attempt-advwebtech-ude.herokuapp.com/");`

comment line.4

`export const UrlContext = createContext("http://localhost:3000/");`


3. Seperate 'client' folder from main project folder to be deployed seperately as in the following guide


4. follow the guide [Deploy web app to Heroku](https://www.youtube.com/playlist?list=PLpSK06odCvYdSyGkWmc-AdqRc3zmiHPCc), mainly you need to do as follows:
* Deploy backend app to heroku after pushing it to github. Please follow the steps in the upmentioned guide
* Create an account on mLab, currently migrated to mongoDB Atlas, make sure to name database and collection as written in the code, and finally connect backend app to mLab as explained in the upmentioned guide
* Push client file to a new github repository and deploy it to heroku. Please follow the steps in the upmentioned guide and __note that here you don't need to change url in axios part as you did that on step number 2__

5. On file `package.json` make sure that proxy value is equal to the url of the deployed frontend app on heroku

# Group members
> <ul><li>Baohui Deng</li> <li>Tannaz Vahidi</li> <li>Amr Shakhshir</li> <li>Hesamoddin Heidarzadeh</li></ul>

# Dataset source
P . Cortez and A. Silva. [Using Data Mining to Predict Secondary School Student Performance.](https://archive.ics.uci.edu/ml/datasets/student+performance#) In A. Brito and J. Teixeira Eds., P roc eedings of 5 th FUture BUsiness
TEChnology Conference (FUBUTEC 2008 ) pp. 5 12 , Porto, Portugal, April, 2008 , EUROSIS, ISBN 978 9077381 39 7