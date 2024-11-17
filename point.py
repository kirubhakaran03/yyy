from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model
with open("./linear_regression_model3.pkl", "rb") as model_file:
    clf = pickle.load(model_file)

@app.route("/predict", methods=['POST'])
def prediction():
    # Get the JSON request data
    mark_req = request.get_json()

    # Extract and convert input features from the JSON data
    try:
        Serial = float(mark_req.get('SerialNo.', 0))
        GRE = float(mark_req.get('GREScore', 0))
        TOEFL = float(mark_req.get('TOEFLScore', 0))
        University = float(mark_req.get('UniversityRating', 0))
        SOP = float(mark_req.get('SOP', 0))
        LOR = float(mark_req.get('LOR', 0))
        CGPA = float(mark_req.get('CGPA', 0))
        Research = float(mark_req.get('Research', 0))
    except (TypeError, ValueError) as e:
        return jsonify({"error": "Invalid input data"}), 400

    # Predict using the model
    result = clf.predict([[Serial, GRE, TOEFL, University, SOP, LOR, CGPA, Research]])

    # Classify based on prediction threshold
    pred = "Rejected" if result > 0.5 else "Approved"

    # Return the prediction result as JSON
    return jsonify({"loan_approval_status": pred})

if __name__ == '__main__':
    app.run(debug=True)
