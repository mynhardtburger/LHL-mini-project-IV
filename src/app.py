# import Flask and jsonify
import json

import joblib
import numpy.typing as npt
import pandas as pd
from flask import Flask, jsonify, request

# import Resource, Api and reqparser
from flask_restful import Api, Resource
from sklearn.pipeline import Pipeline

from custom_transformers import CreateTotalIncome

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
api = Api(app)

model: Pipeline = joblib.load(open("./src/model.joblib", "rb"))


@app.route("/")
def hello_world():
    return "Submit the JSON object to the /scoring endpoint to receive the predicted result"


class Scoring(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).T

        probabilities: npt.NDArray = model.predict_proba(df)
        result: npt.NDArray = model.predict(df)

        res = {
            "Loan_Status": int(result[0]),
            "Certainty": float(probabilities[0][result[0]]),
        }

        return jsonify(res)


api.add_resource(Scoring, "/loan_status")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
