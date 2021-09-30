# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 03:26:15 2021

@author: hp
"""
from flask import Flask, render_template, request
import numpy as np
import pickle
import sklearn
import requests
app = Flask(__name__)
loaded = pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('trying.html')
# prediction function

@app.route('/result', methods = ['POST'])
def result():
	if request.method == 'POST':
           print(request.form.get('Gender'))
           print(request.form.get('Age'))
           print(request.form.get('Height'))
           print(request.form.get('Weight'))
           print(request.form.get('Duration'))
           print(request.form.get('Heart_Rate'))
           print(request.form.get('Body_Temp'))
           
           Gender=int(request.form['Gender'])
           Age=int(request.form['Age'])
           Height=int(request.form['Height'])
           Weight=int(request.form['Weight'])
           Duration=int(request.form['Duration'])
           Heart_Rate=int(request.form['Heart_Rate'])
           Body_Temp=float(request.form['Body_Temp'])
           to_predict_list = [Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp]
           #to_predict = np.array(to_predict_list).reshape(1,7)
           #int_features = [int(x) for x in request.form.values()]
           final_features = np.array(to_predict_list).reshape(1,-1)
           result = loaded.predict(final_features)	
                 
	return render_template("trying.html",prediction="Calories Burnt{} ".format(result))
    
if __name__=='__main__':
     app.run(debug=True)