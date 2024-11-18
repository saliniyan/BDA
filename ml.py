import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data (example: logs_df.csv)
log_df = pd.read_csv('logs_df(ml).csv')

# Count of target values (Status Code) using pandas
status_code_counts = log_df['Status Code'].value_counts()
print(status_code_counts)
