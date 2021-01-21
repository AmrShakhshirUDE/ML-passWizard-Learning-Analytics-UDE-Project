from flask import Flask
from flask_pymongo import PyMongo
from bson.json_util import dumps
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config["MONGO_URI"] = "mongodb://localhost:27017/ladb"
mongo = PyMongo(app)

@app.route("/alldata")
def home_page():
    df = mongo.db.passwizardfe.find()
    resp = dumps(df)
    return resp