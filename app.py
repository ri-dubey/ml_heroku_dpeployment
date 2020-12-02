import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('/Users/riteshdubey/Documents/deployment/salary_prediction/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    exp = request.form.get('experience')
    test_score = request.form.get('test_score')
    inter_score = request.form.get('interview_score')

    int_features1=[int(exp), int(test_score), int(inter_score)]

    int_features = [int(x) for x in request.form.values()]

    final_features1=[np.array(int_features1)]

    final_features = [np.array(int_features)]

    prediction1 = model.predict(final_features1)

    prediction = model.predict(final_features)

    output1 = round(prediction1[0],2)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {} or $ {}'.format(output,output1))


if __name__ == "_main_":
    app.run(debug=True)