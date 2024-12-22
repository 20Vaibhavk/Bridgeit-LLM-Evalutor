from flask import Flask, request, jsonify
from evaluation import evaluate_llm_output
from retrieval_eval import evaluate_retrieval
from task_training import fine_tune_task_model
from werkzeug.utils import secure_filename
import os

# Initialize Flask app
app = Flask(__name__)

# Define the direct datasets directory path
datasets_dir = "D:/Saee/CSE/Projects/LLM-Evaluater/backend/datasets"

# Ensure the datasets directory exists
if not os.path.exists(datasets_dir):
    print(f"Creating datasets directory at {datasets_dir}")
    os.makedirs(datasets_dir, exist_ok=True)
else:
    print(f"Datasets directory already exists at {datasets_dir}")

# Route: Upload Dataset
@app.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    # Check if 'dataset' exists in request files
    if 'dataset' not in request.files:
        return jsonify({"error": "No file uploaded, missing 'dataset' field"}), 400

    file = request.files['dataset']

    # Get the filename directly without secure_filename
    filename = file.filename
    if not filename:
        return jsonify({"error": "Invalid file name, file is empty"}), 400

    # Define the full file path
    file_path = os.path.join(datasets_dir, filename)

    # Debugging statement to check the path and file details
    print(f"Saving file to {file_path}")
    print(f"File details: {file.filename}, {file.mimetype}")

    try:
        # Attempt to save the file
        file.save(file_path)
        return jsonify({"message": "Dataset uploaded successfully", "path": file_path})
    except Exception as e:
        # Provide more specific error message
        print(f"Error saving file: {str(e)}")
        return jsonify({"error": f"Error saving file: {str(e)}"}), 500


# Route: Evaluate LLM Output
@app.route('/evaluate-output', methods=['POST'])
def evaluate_output():
    data = request.get_json()
    if 'responses' not in data or 'ground_truths' not in data:
        return jsonify({"error": "Missing 'responses' or 'ground_truths'"}), 400

    results = evaluate_llm_output(data['responses'], data['ground_truths'])
    return jsonify(results)

# Route: Evaluate Retrieval
@app.route('/evaluate-retrieval', methods=['POST'])
def evaluate_retrieval_endpoint():
    data = request.get_json()
    if 'queries' not in data or 'retrieved_docs' not in data or 'ground_truth_docs' not in data:
        return jsonify({"error": "Missing required fields"}), 400

    results = evaluate_retrieval(data['queries'], data['retrieved_docs'], data['ground_truth_docs'])
    return jsonify(results)

# Route: Fine-Tune LLM on a Task
@app.route('/fine-tune', methods=['POST'])
def fine_tune():
    data = request.get_json()
    if 'task_name' not in data or 'training_data' not in data:
        return jsonify({"error": "Missing 'task_name' or 'training_data'"}), 400

    results = fine_tune_task_model(data['task_name'], data['training_data'])
    return jsonify(results)

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
