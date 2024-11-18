from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("ClassificationExamples").getOrCreate()

# Load your data (example: logs_df.csv)
log_df = spark.read.csv('logs_df(ml).csv', header=True, inferSchema=True)

# Feature engineering (assemble features)
feature_cols = ['Method', 'Endpoint', 'Protocol', 'Content Size', 'No of Requests']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
log_df = assembler.transform(log_df)

# Split into train and test sets
train_df, test_df = log_df.randomSplit([0.7, 0.3], seed=42)

# Standardize features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(train_df)
train_df = scaler_model.transform(train_df)
test_df = scaler_model.transform(test_df)

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(labelCol="Status Code", featuresCol="scaledFeatures"),
    "Decision Tree": DecisionTreeClassifier(labelCol="Status Code", featuresCol="scaledFeatures"),
    "Random Forest": RandomForestClassifier(labelCol="Status Code", featuresCol="scaledFeatures"),
}

# Train each classifier, evaluate accuracy, save the models, and print results
evaluator = MulticlassClassificationEvaluator(labelCol="Status Code", predictionCol="prediction", metricName="accuracy")

for name, clf in classifiers.items():
    # Train the model
    model = clf.fit(train_df)
    
    # Save the trained model
    model_path = f"/home/saliniyan/Documents/BDA/{name.replace(' ', '_')}_model1"
    model.save(model_path)
    print(f"{name} model saved at {model_path}")
    
    # Save the scaler model
    scaler_path = f"/home/saliniyan/Documents/BDA/{name.replace(' ', '_')}_scaler1"
    scaler_model.save(scaler_path)
    print(f"{name} scaler model saved at {scaler_path}")
    
    # Make predictions
    predictions = model.transform(test_df)
    
    # Evaluate accuracy
    accuracy = evaluator.evaluate(predictions)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
