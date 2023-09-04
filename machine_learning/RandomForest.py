from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate

from MachineLearningUtil import MLUtil


ml = MLUtil("random_forest")
print('\nML:', ml.name)

ml.define_data_for_machine_learning()
X_train, X_test, y_train, y_test = train_test_split(ml.x, ml.y, test_size = 0.3, train_size=0.7, random_state=0)
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)


# Cross-validation
scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
results = cross_validate(model, ml.x, ml.y, cv=10, scoring=scoring)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

ml.predictions_and_csv(y_test, y_pred, results)
