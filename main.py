import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("heart.csv", sep = ",")

le = LabelEncoder()

df['Sex'], df['ChestPainType'], df['RestingECG'], df['ExerciseAngina'], df['ST_Slope'] = le.fit_transform(df['Sex']), le.fit_transform(df['ChestPainType']), le.fit_transform(df['RestingECG']), le.fit_transform(df['ExerciseAngina']), le.fit_transform(df['ST_Slope'])

X = df.drop("HeartDisease", 1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(classification_report(y_test, y_pred))

filename = 'class_model.sav'
pickle.dump(rfc, open(filename, 'wb'))


# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, y_test)
# print(result)