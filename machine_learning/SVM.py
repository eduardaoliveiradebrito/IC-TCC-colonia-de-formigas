from sklearn import svm
from sklearn.model_selection import train_test_split, cross_validate

from MachineLearningUtil import MLUtil


ml = MLUtil("svm")
print('\nML:', ml.name)

ml.define_data_for_machine_learning_SVM()
X_train, X_test, y_train, y_test = train_test_split(ml.x, ml.y, test_size=0.3)
model = svm.SVC()


# Cross-validation
scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
results = cross_validate(model, ml.x, ml.y, cv=10, scoring=scoring)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

ml.predictions_and_csv(y_test, y_pred, results)
