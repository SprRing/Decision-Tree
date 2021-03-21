
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics # Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing

### load dataset
col_names = ['Outlook', 'Temperature', 'Humidity', 'Windy', 'Play']
pima = pd.read_csv("/Users/yenshou/Desktop/Data mining/DTs.csv", header=0, names=col_names)
df= pima.copy()

### Preprocessing
df_dum = pd.get_dummies(df['Outlook'])
df_new=pd.concat([df_dum,df[['Temperature', 'Humidity', 'Windy', 'Play']]],axis=1)
le = preprocessing.LabelEncoder()
df_new['Windy'] = le.fit_transform(df_new['Windy'])

print(df_new)
# split dataset in features and target variable
feature_cols = ['Overcast', 'Rainy', 'Sunny', 'Temperature', 'Humidity', 'Windy']
X = df_new[feature_cols] # Features
y = df_new.Play # Target variable

# Split dataset into training set and test set
X_train, X_test = X.loc[0:13], X.loc[14]
y_train, y_test = y.loc[0:13], y.loc[14]

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy")

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict([X_test])

# Model Accuracy, how often is the classifier correct?

print("using entropy", y_pred)
