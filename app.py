from flask import Flask, request, render_template
import numpy as np
import pickle

# Load models and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standardscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Extract and convert form values
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        features = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(features).reshape(1, -1)

        # Apply transformations
        mx_scaled = mx.transform(single_pred)
        final_input = sc.transform(mx_scaled)
        prediction = model.predict(final_input)

        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
            8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        crop = crop_dict.get(prediction[0], "Unknown crop")
        result = f"🌾 Recommended Crop: {crop}"

    except Exception as e:
        result = f" Error: {str(e)}"

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=10000)