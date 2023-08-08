import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_validate
import warnings



data = pd.read_csv('C:/Users/maria.oliveira/Documents/workspace/projetos/IC-TCC-colonia-de-formigas/bases/Banana/banana.csv', sep=',')
variables = ['At1','At2']

x = data[variables]
y = data['Class']


# Count of classes that do not meet the number of cross folds
class_count = data['Class'].value_counts()

num_folds_cross = 5
# Classes with less than num_folds_cross of samples
minor_classes_num_folds= class_count[class_count < num_folds_cross].index.tolist()

# Filter classes with less than num_folds_cross of samples
filtered_data = data[~data['Class'].isin(minor_classes_num_folds)]

x = filtered_data[variables]
y = filtered_data['Class']


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, train_size=0.7, random_state=0)
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)


# Suppress specific warnings related to undefined metrics (Warnings of division by 0 occurred, because for some classes it is not possible to calculate f1, because the precision was 0)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

# Cross-validation
scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
results = cross_validate(model, x, y, cv=num_folds_cross, scoring=scoring)  # cv = número de folds

# Fit the model using all training data
model.fit(X_train, y_train)

# Make predictions from test data
y_pred = model.predict(X_test)


# Evaluation metrics on test data, by class where the metric is calculated for each class individually and returned as a list of values instead of a single aggregated value.
print('\nEvaluation metrics in the data - By class:')
print('Acurácia:', accuracy_score(y_test, y_pred) * 100)
print('F-Measure:', f1_score(y_test, y_pred, pos_label=1, average=None) * 100)
print('Precisão:', precision_score(y_test, y_pred, pos_label=1, average=None) * 100)
print('Recall:', recall_score(y_test, y_pred, pos_label=1, average=None) * 100)

# Evaluation metrics on test data, by aggregate where the metric is calculated for each class individually and then the values are simply averaged.
# In this case, all classes have the same importance and contribute equally to the aggregate result.
print('\nEvaluation metrics in the data - By Aggregate:')
print('Acurácia:', accuracy_score(y_test, y_pred) * 100)
print('F-Measure:', f1_score(y_test, y_pred, pos_label=1, average='macro') * 100)
print('Precisão:', precision_score(y_test, y_pred, pos_label=1, average='macro') * 100)
print('Recall:', recall_score(y_test, y_pred, pos_label=1, average='macro') * 100)

# Restore default warning settings
warnings.filterwarnings("default", category=UserWarning, module="sklearn.metrics")


# Mean and standard deviation of cross-validation scores
print('\n\nCross validation - Average of metrics:')
print('Acurácia média:', results['test_accuracy'].mean())
print('F-Measure média:', results['test_f1_macro'].mean())
print('Precisão média:', results['test_precision_macro'].mean())
print('Recall média:', results['test_recall_macro'].mean())

print('\nCross-validation - Standard deviation of metrics:')
print('Acurácia desvio padrão:', results['test_accuracy'].std())
print('F-Measure desvio padrão:', results['test_f1_macro'].std())
print('Precisão desvio padrão:', results['test_precision_macro'].std())
print('Recall desvio padrão:', results['test_recall_macro'].std())
