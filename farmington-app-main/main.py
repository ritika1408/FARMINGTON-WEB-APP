import smtplib

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, Markup
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image

import requests

# email and password to send enquiries to the contact us page
OWN_EMAIL = 'rpmfarmington@gmail.com'
OWN_PASSWORD = 'rgfednyqhatcyfhq'

# loading the crop recommendation model
crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# ------------------------------------------------------------------------------------------------#
# custom functions
# function to calculate the values of temperature and humidity in crop recommendation

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404":
        # store the value of "main" key in variable y
        # if __main__ == '__response.json() __':
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


app = Flask(__name__)
app.debug = True


# home route
@app.route("/")
def home():
    title = 'FARMINGTON'
    return render_template("index.html", title=title)


# route to crop recommendation
@app.route('/crop')
def crop_recommendation():
    title = 'Farmington - Crop Recommendation'
    return render_template('crop.html', title=title)


# route to fertilizer suggestion
@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Farmington - Fertilizer Recommendation'
    return render_template('fertilizer.html', title=title)


# route to disease detection
@app.route('/disease')
def disease_detection():
    title = 'Farmington - Disease Detection'
    return render_template('disease.html', title=title)


# Render the result pages of each and every system


# Handle Post Requests -- Crop prediction  Result Page
@app.route('/crop_predict', methods=['POST'])
def crop_prediction():
    title = 'Farmington - Crop Recommendation Result'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorus'])
        K = int(request.form['potassium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)


# Handle Post Requests -- Fertilizer Recommendation Result Page
@app.route('/fertilizer_predict', methods=['POST'])
def fert_recommend():
    title = "Farmington - Fertilizer Suggestion"

    # requesting data from the form
    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorus'])
    K = int(request.form['potassium'])
    # ph = float(request.form['ph'])
    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    # finding out the variation in the given input and the model's actual values
    n = nr - N
    p = pr - P
    k = kr - K

    # putting the absolute values of the differences in a dictionary
    temp = {abs(n): 'N', abs(p): 'P', abs(k): 'K'}
    max_value = temp[max(temp.keys())]

    if max_value == 'N':
        if n < 0:
            key = "NHigh"
        else:
            key = "Nlow"

    elif max_value == 'P':
        if p < 0:
            key = "PHigh"
        else:
            key = "Plow"

    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = 'Klow'

    response = Markup(str(fertilizer_dic[key]))
    return render_template('fertilizer-result.html', recommendation=response, title=title)


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Farmington - Disease Detection'

    # if request.method == 'POST':
    #     if 'file' not in request.files:
    #         return redirect(request.url)
    #     file = request.files.get('file')
    #     if not file:
    #         return render_template('disease.html', title=title)
    #     try:
    #         img = file.read()
    #
    #         prediction = predict_image(img)
    #
    #         prediction = Markup(str(disease_dic[prediction]))
    #         return render_template('disease-result.html', prediction=prediction, title=title)
    #     except:
    #         pass
    # return render_template('disease.html', title=title)


# Navbar links

@app.route('/aboutus')
def about():
    title = 'Farmington - About Us'
    return render_template('about.html', title=title)


@app.route('/services')
def services():
    title = 'Farmington - Our Services'
    return render_template('services.html', title=title)


@app.route('/faqs')
def faqs_ask():
    title = 'Farmington - FAQ'
    return render_template('faqs.html', title=title)


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    title = 'Farmington - CONTACT US '
    if request.method == "POST":
        data = request.form
        data = request.form
        send_email(data["fname"], data["email"], data["phone"], data["message"])
        return render_template("contact.html", msg_sent=True, title=title)
    return render_template("contact.html", msg_sent=False, title=title)


def send_email(fname, email, phone, message):
    email_message = f"Subject: New Message \n\nFull Name: {fname}\nEmail:{email}\nPhone:{phone}\nMessage:{message}"
    with smtplib.SMTP("smtp.gmail.com") as connection:
        connection.starttls()
        connection.login(OWN_EMAIL, OWN_PASSWORD)
        connection.sendmail(OWN_EMAIL, OWN_EMAIL, email_message)


if __name__ == '__main__':
    app.run()
