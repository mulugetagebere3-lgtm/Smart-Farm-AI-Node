from flask import Flask, render_template_string
import joblib
import pandas as pd
import os

app = Flask(__name__)

# 1. AI Model Load (ነቲ ዝሰልጠነ ሞዴል ምጽዓን)
model_path = 'xgb_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

@app.route('/')
def index():
    try:
        # 2. ዳታ ካብቲ CSV ነንብብ
        data = pd.read_csv('test.csv')
        last_row = data.tail(1).drop(columns=['id'], errors='ignore')
        
        # 3. AI Prediction (እቲ ሞዴል ውሳነ ይህብ)
        X_encoded = pd.get_dummies(last_row)
        expected_columns = model.feature_names_in_
        for col in expected_columns:
            if col not in X_encoded.columns: X_encoded[col] = 0
            
        prediction = model.predict(X_encoded[expected_columns])[0]

        # 4. ዳታ ንግራፍ (Moisture, Temp, Humidity)
        moisture = float(last_row['Soil_Moisture'].iloc[0])
        temp = float(last_row['Temperature_C'].iloc[0])
        hum = float(last_row['Humidity'].iloc[0])

        status = "⚠️ WATER NEEDED" if prediction == 1 else "✅ SOIL IS GOOD"
        color = "#e74c3c" if prediction == 1 else "#2ecc71"

    except Exception as e:
        return f"<h1>Error: {e}</h1>"

    # 5. Dashboard UI (HTML & Chart.js)
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="refresh" content="5">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: sans-serif; background: #1a1a2e; color: white; text-align: center; }}
            .card {{ background: #16213e; max-width: 500px; margin: 20px auto; padding: 20px; border-radius: 20px; border: 1px solid #34495e; }}
            .status {{ background: {color}; padding: 20px; border-radius: 10px; font-size: 24px; font-weight: bold; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Farm AI Node</h1>
            <div class="status">{status}</div>
            <canvas id="myChart"></canvas>
            <p style="color: #bdc3c7;">Edge Computing: Raspberry Pi 5</p>
        </div>
        <script>
            const ctx = document.getElementById('myChart').getContext('2d');
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: ['Moisture', 'Temp (°C)', 'Humidity'],
                    datasets: [{{
                        label: 'Real-time Metrics',
                        data: [{moisture}, {temp}, {hum}],
                        backgroundColor: ['#3498db', '#f1c40f', '#9b59b6']
                    }}]
                }},
                options: {{ scales: {{ y: {{ beginAtZero: true, grid: {{ color: '#333' }} }} }} }}
            }});
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
