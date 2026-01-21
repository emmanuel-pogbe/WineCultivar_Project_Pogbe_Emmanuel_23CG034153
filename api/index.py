"""
Wine Cultivar Origin Prediction System - Flask Web Application
===============================================================
A web-based GUI for predicting wine cultivar origin based on 
chemical properties using a trained machine learning model.
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

# Initialize Flask application
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ============================================================================
# LOAD TRAINED MODEL AND SCALER
# ============================================================================

def load_model_and_scaler():
    """Load the trained model and scaler from disk."""
    try:
        # Load the trained model
        dir = "model"
        with open(f'{dir}/wine_cultivar_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load the scaler
        with open(f'{dir}/wine_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        print("✓ Model and scaler loaded successfully!")
        return model, scaler
    except FileNotFoundError as e:
        print(f"Error: Could not find model files - {e}")
        return None, None

# Load model and scaler globally
MODEL, SCALER = load_model_and_scaler()

# Define selected features (in order)
SELECTED_FEATURES = [
    'alcohol', 'malic_acid', 'total_phenols',
    'flavanoids', 'color_intensity', 'proline'
]

# Define cultivar names
CULTIVAR_NAMES = {
    1: 'Cultivar 1',
    2: 'Cultivar 2',
    3: 'Cultivar 3'
}

# Feature information for UI
FEATURE_INFO = {
    'alcohol': {
        'label': 'Alcohol Content (%)',
        'min': 11.0,
        'max': 15.0,
        'step': 0.1,
        'placeholder': '12.5',
        'help': 'Alcohol percentage in wine (11.0 - 15.0%)'
    },
    'malic_acid': {
        'label': 'Malic Acid (g/L)',
        'min': 0.5,
        'max': 6.0,
        'step': 0.1,
        'placeholder': '2.5',
        'help': 'Malic acid concentration (0.5 - 6.0 g/L)'
    },
    'total_phenols': {
        'label': 'Total Phenols (mg/L)',
        'min': 1.0,
        'max': 4.0,
        'step': 0.1,
        'placeholder': '2.3',
        'help': 'Total phenolic compounds (1.0 - 4.0 mg/L)'
    },
    'flavanoids': {
        'label': 'Flavanoids (mg/L)',
        'min': 0.3,
        'max': 5.5,
        'step': 0.1,
        'placeholder': '2.0',
        'help': 'Flavanoid compounds (0.3 - 5.5 mg/L)'
    },
    'color_intensity': {
        'label': 'Color Intensity',
        'min': 0.1,
        'max': 0.7,
        'step': 0.01,
        'placeholder': '0.35',
        'help': 'Wine color intensity (0.1 - 0.7 AU)'
    },
    'proline': {
        'label': 'Proline (mg/L)',
        'min': 250,
        'max': 1700,
        'step': 10,
        'placeholder': '750',
        'help': 'Proline amino acid concentration (250 - 1700 mg/L)'
    }
}

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Render the main application page."""
    return render_template('index.html', features=FEATURE_INFO)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle prediction requests from the frontend."""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate that all features are provided
        missing_features = [f for f in SELECTED_FEATURES if f not in data]
        if missing_features:
            return jsonify({
                'error': f'Missing features: {", ".join(missing_features)}',
                'success': False
            }), 400
        
        # Extract feature values in correct order
        feature_values = []
        for feature in SELECTED_FEATURES:
            try:
                value = float(data[feature])
                feature_values.append(value)
            except (ValueError, TypeError):
                return jsonify({
                    'error': f'Invalid value for {feature}: must be a number',
                    'success': False
                }), 400
        
        # Convert to numpy array and reshape
        features = np.array(feature_values).reshape(1, -1)
        
        # Scale the features
        features_scaled = SCALER.transform(features)
        
        # Make prediction
        prediction = MODEL.predict(features_scaled)[0]
        probabilities = MODEL.predict_proba(features_scaled)[0]
        
        # Prepare response
        response = {
            'success': True,
            'predicted_cultivar': CULTIVAR_NAMES[prediction],
            'cultivar_id': int(prediction),
            'confidence': float(max(probabilities) * 100),
            'probabilities': {
                CULTIVAR_NAMES[i+1]: float(prob * 100) 
                for i, prob in enumerate(probabilities)
            },
            'input_features': {
                feature: float(value)
                for feature, value in zip(SELECTED_FEATURES, feature_values)
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'success': False
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Return model information."""
    return jsonify({
        'algorithm': 'Random Forest Classifier',
        'features': SELECTED_FEATURES,
        'cultivars': list(CULTIVAR_NAMES.values()),
        'test_accuracy': 0.9722,
        'test_accuracy_percent': '97.22%'
    }), 200

@app.route('/api/example-data', methods=['GET'])
def example_data():
    """Return example wine data for demonstration."""
    examples = {
        'sample_1': {
            'name': 'Sample Wine 1',
            'description': 'High alcohol, high flavanoids',
            'data': {
                'alcohol': 14.23,
                'malic_acid': 1.71,
                'total_phenols': 2.8,
                'flavanoids': 3.06,
                'color_intensity': 5.64,
                'proline': 1065
            }
        },
        'sample_2': {
            'name': 'Sample Wine 2',
            'description': 'Medium alcohol, medium characteristics',
            'data': {
                'alcohol': 13.2,
                'malic_acid': 1.78,
                'total_phenols': 2.65,
                'flavanoids': 2.76,
                'color_intensity': 4.38,
                'proline': 1050
            }
        },
        'sample_3': {
            'name': 'Sample Wine 3',
            'description': 'Lower alcohol, different profile',
            'data': {
                'alcohol': 12.37,
                'malic_acid': 1.21,
                'total_phenols': 2.56,
                'flavanoids': 2.67,
                'color_intensity': 3.04,
                'proline': 985
            }
        }
    }
    return jsonify(examples), 200

@app.route('/api/reset', methods=['POST'])
def reset_form():
    """Return default values for form reset."""
    default_values = {
        'alcohol': '',
        'malic_acid': '',
        'total_phenols': '',
        'flavanoids': '',
        'color_intensity': '',
        'proline': ''
    }
    return jsonify(default_values), 200

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Page not found', 'success': False}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error', 'success': False}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Check if model is loaded
    if MODEL is None or SCALER is None:
        print("ERROR: Could not load model or scaler. Make sure the .pkl files are in the model/ directory.")
        exit(1)
    
    print("\n" + "="*70)
    print("WINE CULTIVAR PREDICTION SYSTEM - WEB APPLICATION")
    print("="*70)
    print("\n✓ Flask application initialized successfully!")
    print("✓ Model and scaler loaded!")
    print("\n📊 Model Information:")
    print(f"   - Algorithm: Random Forest Classifier")
    print(f"   - Features: {len(SELECTED_FEATURES)} (alcohol, malic_acid, total_phenols,")
    print(f"                          flavanoids, color_intensity, proline)")
    print(f"   - Classes: 3 (Cultivar 1, 2, 3)")
    print(f"   - Test Accuracy: 97.22%")
    print("\n🌐 Web Server:")
    print("   - Starting Flask server on http://localhost:5000")
    print("   - Open your browser and navigate to http://localhost:5000")
    print("\n💡 Features:")
    print("   - Input wine chemical properties")
    print("   - Get instant cultivar predictions")
    print("   - View confidence scores")
    print("   - See probability distribution")
    print("\n⚠️  Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    # Run the Flask application
    app.run(debug=True, host='localhost', port=5000)
