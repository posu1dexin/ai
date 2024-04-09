from flask import Flask, render_template, request, jsonify
import json
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtain token, project ID, model name, and image filename from the request
    token = request.form['token']
    project_id = request.form['project_id']
    model = request.form['model']
    image_url = request.form['image_url']
    
    payload = json.dumps({"url": image_url})
    # Prepare headers with the token and content type
    headers = {"X-Auth-token": token, "Content-Type": "application/json"}

    # Make a POST request to the API endpoint with the image data and headers
    response = requests.post('https://platform.sentisight.ai/api/predict/{}/{}/'.format(project_id, model), headers=headers, data=payload)

    if response.status_code == 200:
        # Process and return the predictions
        return render_template('results.html', image_url=image_url, results=response.json())
    else:
        # Handle error response
        return f'Error occurred with REST API. Status code: {response.status_code}\nError message: {response.text}'

if __name__ == '__main__':
    app.run(debug=True)
