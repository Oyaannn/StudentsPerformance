import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load data
df = pd.read_csv("StudentsPerformance.csv")

# Hitung nilai rata-rata
df["average score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)

# Label kelulusan: >= 80 = Lulus, < 80 = Tidak Lulus
df["graduation"] = df["average score"].apply(lambda x: "Lulus" if x >= 75 else "Tidak Lulus")

# Label encoding untuk fitur kategorik
le = LabelEncoder()
for column in ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]:
    df[column] = le.fit_transform(df[column])

# Fitur dan label
X = df[["gender", "race/ethnicity", "parental level of education", "lunch",
        "test preparation course", "math score", "reading score", "writing score"]]
y = df["graduation"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# Simpan model dan scaler
joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")