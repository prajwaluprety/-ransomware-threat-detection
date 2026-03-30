import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('ransomware_data.csv')

# Print basic info
print(f"Total samples: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()
print("Label breakdown:")
print(df['label'].value_counts())
print()
print("First 5 rows:")
print(df.head())

# Draw a chart
df['label'].value_counts().plot(kind='bar', color=['#E24B4A','#1D9E75'])
plt.title('Ransomware vs Benign samples')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()
print("Chart saved as class_distribution.png")

