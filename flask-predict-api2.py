import pickle

from flask import Flask, request

from sklearn.externals import joblib

from flasgger import Swagger

import numpy as np

import pandas as pd



app = Flask(__name__)

swagger = Swagger(app)



with open('rf.pkl', 'rb') as model_file:

    model = pickle.load(model_file)



@app.route('/predict')

def predict_iris():

    """Example endpoint returning a list of colors by value

        This is using docstrings for specifications.

        ---

        tags:

          - Iris Prediction API Input values

        parameters:

          - name: s_length

            in: query

            type: number

            required: true

          - name: s_width

            in: query

            type: number

            required: true

          - name: p_length

            in: query

            type: number

            required: true

          - name: p_width

            in: query

            type: number

            required: true

        definitions:

          value:

            type: object

            properties:

              value_name:

                type: string

                items:

                  $ref: '#/definitions/Color'

          Color:

            type: string

        responses:

          200:

            description: OK

            schema:

              $ref: '#/definitions/value'

    """

    s_length = (request.args.get("s_length"))

    s_width = (request.args.get("s_width"))

    p_length = (request.args.get("p_length"))

    p_width = (request.args.get("p_width"))

    prediction = model.predict(np.array([[s_length, s_width,

                                        p_length, p_width]]))

    return str(prediction)



@app.route('/predict_file', methods=["POST"])

def predict_iris_file():

    """Example endpoint returning a list of colors by value

    This is using docstrings for specifications.

    ---

    tags:

      - Iris Prediction API Upload .csv file

    parameters:

      - name: input_file

        in: formData

        type: file

        required: true

    definitions:

      value:

        type: object

        properties:

          value_name:

            type: string

            items:

              $ref: '#/definitions/Color'

      Color:

        type: string

    responses:

      200:

        description: OK

        schema:

          $ref: '#/definitions/value'

    """

    input_data = pd.read_csv(request.files.get("input_file"), header=None)

    prediction = model.predict(input_data)

    return str(list(prediction))



if __name__== '__main__':

    app.run(host='0.0.0.0', port=5001)
