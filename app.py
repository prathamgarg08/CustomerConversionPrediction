import pickle
from flask import Flask,request,url_for,render_template,app,jsonify
import numpy as np
import pandas as pd

app=Flask(__name__)
rfmodel=pickle.load(open('randomforest.pkl','rb'))
scaling=pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predicter_api',methods=['POST'])
def predicter_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaling.transform(np.array(list(data.values())).reshape(1,-1))
    output=rfmodel.predict(new_data)
    print(output)
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaling.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=rfmodel.predict(final_input)[0]
    return render_template('home.html',prediction_text='Customer Conversion Rate is {}'.format(output))



if __name__=='__main__':
    app.run(debug=True)