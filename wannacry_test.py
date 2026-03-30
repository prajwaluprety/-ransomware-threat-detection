import pandas as pd
import numpy as np
import joblib

# Load the saved model
model = joblib.load('ransomware_detector.pkl')
le = joblib.load('label_encoder.pkl')

# Load original data to get column structure
df = pd.read_csv('ransomware_data.csv')
X = df.drop('label', axis=1)

# Create WannaCry simulation profile
# WannaCry is known for: massive file operations,
# high entropy, lots of renames, heavy network activity
wannacry = pd.DataFrame([X.mean()], columns=X.columns)
wannacry['file_count']       = 4500   # opens thousands of files
wannacry['write_operations'] = 2800   # rewrites everything
wannacry['cpu_usage']        = 95.0   # maxes out CPU encrypting
wannacry['entropy']          = 7.9    # encrypted files look random
wannacry['network_calls']    = 480    # scans network for more victims
wannacry['file_renames']     = 1800   # renames files to .WNCRY
wannacry['registry_changes'] = 450    # modifies Windows registry
wannacry['api_calls']        = 4800   # thousands of system calls

# Run through the detector
prediction = model.predict(wannacry)[0]
probability = model.predict_proba(wannacry)[0]
label = le.inverse_transform([prediction])[0]
confidence = round(max(probability) * 100, 1)

print("=" * 50)
print("   RANSOMWARE DETECTION SYSTEM")
print("=" * 50)
print(f"   Sample:     WannaCry simulation")
print(f"   Verdict:    {label.upper()}")
print(f"   Confidence: {confidence}%")
print("=" * 50)
if label == 'ransomware':
    print("   ALERT: Ransomware detected!")
    print("   Action: Isolate system immediately")
else:
    print("   System appears safe")
print("=" * 50)