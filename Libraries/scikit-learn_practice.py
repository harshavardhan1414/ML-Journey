# first in this file we will see few models then training and testing process then model evalution and preprocessing sacling

#--> 1.linear_regression.py

import numpy as np 
from sklearn.linear_model import LinearRegression
# here i will take house size vs price
X = np.array([[500], [1000], [1500], [2000], [2500]])
y = np.array([150, 300, 450, 600, 750])
model = LinearRegression()
model.fit(X, y)
prediction = model.predict([[1800]])
print( prediction)


#---> train_test_split
from sklearn.model_selection import train_test_split
import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([10, 20, 30, 40, 50, 60])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=2
) # we can use random_state value 42 it will be best for industry case
print( X_train) # this both x and y gives data under train and test
print( X_test)


#----> model_evaluation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1])
model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)
print( accuracy_score(y, y_pred))
print( confusion_matrix(y, y_pred))
# like here in case of classification to know the accuracy we use accuracy_score

#-----> preprocessing_scaling
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[1000], [1500], [2000], [2500]])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print( X)
print( X_scaled)
# here we use standard scaler as make values in feature cols not deviate more it will autmatically adust the data in nearest values 


#************* example usecase
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X = np.array([[500], [1000], [1500], [2000], [2500], [3000]])
y = np.array([150, 300, 450, 600, 750, 900])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predictions:", y_pred)
print("Actual:", y_test)
print("R2 Score:", r2_score(y_test, y_pred))


# if we come to industry standard use case

#1****pipeline_example
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X = np.array([[1, 200], [2, 300], [3, 400], [4, 500]])
y = np.array([0, 0, 1, 1])
#pipe line creation
pipeline=Pipeline([("scaler",StandardScaler()),("model",LogisticRegression())])
pipeline.fit(X,y)
pred=pipeline.predict([[2.5,450]])
print(pred)
# here,we use pipelines to avoid data leakage


#2**** cross validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
# Cross-validation
scores = cross_val_score(pipeline, X_train, y_train, cv=5)
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)


#3**** grid_search_tuning  here,if we use grid search no need to use cv
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'model__penalty': ['l2']
}

#  gridsearch with crossvalidation
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5 => cross validate,
    scoring='accuracy',
    refit=True,
    n_jobs=-1 =>like how many cores we use -1 for all
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

#  Final Evaluation on Test Set
y_pred = best_model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# *****model_persistence
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

model = LinearRegression()
model.fit(X, y)

# save model
joblib.dump(model, "linear_model.pkl")
#load model
loaded_model = joblib.load("linear_model.pkl")
prediction = loaded_model.predict([[5]])
print("Loaded Model Prediction:", prediction)

## we use this like not to build models again and again
