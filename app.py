from flask import Flask, render_template, request
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.ml.feature import StandardScalerModel, VectorAssembler
from pyspark.sql import SparkSession
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder.appName("DecisionTreePrediction").getOrCreate()

# Load the saved Decision Tree model and scaler
model_path = '/home/saliniyan/Documents/BDA/Decision_Tree_model1'
scaler_path = '/home/saliniyan/Documents/BDA/Decision_Tree_scaler1'

# Load models
dt_model = DecisionTreeClassificationModel.load(model_path)
scaler_model = StandardScalerModel.load(scaler_path)

# Define the route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs from the form
        method = int(request.form['method'])
        endpoint = int(request.form['endpoint'])
        protocol = int(request.form['protocol'])
        content_size = float(request.form['content_size'])
        no_of_requests = int(request.form['no_of_requests'])

        # Prepare the input data
        input_data = np.array([[method, endpoint, protocol, content_size, no_of_requests]])

        # Convert the input data to a Spark DataFrame
        input_df = spark.createDataFrame(input_data, ['Method', 'Endpoint', 'Protocol', 'Content Size', 'No of Requests'])

        # Assemble the features (use the same feature engineering process)
        feature_cols = ['Method', 'Endpoint', 'Protocol', 'Content Size', 'No of Requests']
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        input_df = assembler.transform(input_df)

        # Scale the input data using the loaded scaler
        scaled_input = scaler_model.transform(input_df)

        # Make the prediction
        prediction = dt_model.transform(scaled_input)

        # Get the predicted class
        predicted_class = prediction.select("prediction").collect()[0]["prediction"]

        # Return the result
        return render_template('index.html', prediction=predicted_class)

    except Exception as e:
        return f"Error: {str(e)}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
