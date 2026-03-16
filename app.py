from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved pipeline
with open('california_knn_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Convert JSON to DataFrame (model expects features in specific order/names)
        query_df = pd.DataFrame(data)
        
        # Make prediction
        prediction = model.predict(query_df)
        
        # Return result as JSON
        return jsonify({
            'prediction': prediction.tolist(),
            'unit': '$100,000s'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)