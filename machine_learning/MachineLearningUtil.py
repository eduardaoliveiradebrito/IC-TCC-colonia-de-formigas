import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
import csv


class MLUtil:

  def __init__(self, name):
    self.name = name


  def define_data_for_machine_learning (self):
    data = pd.read_csv('C:/Users/maria.oliveira/Documents/workspace/projetos/IC-TCC-colonia-de-formigas/bases/Banana/banana_Reduzido_paralelo.csv', sep=',')
    variables = ['At1','At2']

    self.x = data[variables]
    self.y = data['Class']


    # Count of classes that do not meet the number of cross folds
    class_count = data['Class'].value_counts()

    num_folds_cross = 10
    # Classes with less than num_folds_cross of samples
    minor_classes_num_folds= class_count[class_count < num_folds_cross].index.tolist()

    # Filter classes with less than num_folds_cross of samples
    filtered_data = data[~data['Class'].isin(minor_classes_num_folds)]

    self.x = filtered_data[variables]
    self.y = filtered_data['Class']


  def define_data_for_machine_learning_SVM (self):
    data = pd.read_csv('C:/Users/maria.oliveira/Documents/workspace/projetos/IC-TCC-colonia-de-formigas/bases/Banana/banana_Reduzido_paralelo.csv', sep=',')

    # Separating features and classes
    self.x = data.drop('Class', axis=1)
    self.y = data['Class']


    # Count of classes that do not meet the number of cross folds
    class_count = data['Class'].value_counts()

    num_folds_cross = 10
    # Classes with less than num_folds_cross of samples
    minor_classes_num_folds= class_count[class_count < num_folds_cross].index.tolist()

    # Filter classes with less than num_folds_cross of samples
    filtered_data = data[~data['Class'].isin(minor_classes_num_folds)]

    self.x = filtered_data.drop('Class', axis=1)
    self.y = filtered_data['Class']


  def print_by_metrics (self, accuracy, f1, precision, recall):
    print('Acurácia:', accuracy)
    print('F-Measure:', f1)
    print('Precisão:', precision)
    print('Recall:', recall)


  def print_by_metrics_cross (self, results):
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


  def create_csv (self, dataCSV, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        for line in dataCSV:
            writer.writerow(line)

    print('\nDados gravados com sucesso no arquivo CSV!')


  def predictions_and_csv (self, y_test, y_pred, results):
    # Suppress specific warnings related to undefined metrics (Warnings of division by 0 occurred, because for some classes it is not possible to calculate f1, because the precision was 0)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

    # Evaluation metrics on test data, by class where the metric is calculated for each class individually and returned as a list of values instead of a single aggregated value.
    print('\nEvaluation metrics in the data - By class:')
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1_class = f1_score(y_test, y_pred, pos_label=1, average=None) * 100
    precision_class = precision_score(y_test, y_pred, pos_label=1, average=None) * 100
    recall_class = recall_score(y_test, y_pred, pos_label=1, average=None) * 100

    self.print_by_metrics(accuracy, f1_class, precision_class, recall_class)

    # Evaluation metrics on test data, by aggregate where the metric is calculated for each class individually and then the values are simply averaged.
    # In this case, all classes have the same importance and contribute equally to the aggregate result.
    print('\nEvaluation metrics in the data - By Aggregate:')
    f1_aggregate = f1_score(y_test, y_pred, pos_label=1, average='macro') * 100
    precision_aggregate = precision_score(y_test, y_pred, pos_label=1, average='macro') * 100
    recall_aggregate = recall_score(y_test, y_pred, pos_label=1, average='macro') * 100

    self.print_by_metrics(accuracy , f1_aggregate , precision_aggregate , recall_aggregate )

    # Restore default warning settings
    warnings.filterwarnings("default", category=UserWarning, module="sklearn.metrics")

    # Mean and standard deviation of cross-validation scores
    self.print_by_metrics_cross(results)


    # Generate CSV with metrics - By class
    dataCSV = [
        ['Acurácia', round(accuracy, 2)],
        ['F-Measure', np.round(f1_class, 2)],
        ['Precisão', np.round(precision_class, 2)],
        ['Recall', np.round(recall_class, 2)]
    ]

    self.create_csv (dataCSV, 'C:/Users/maria.oliveira/Documents/workspace/projetos/IC-TCC-colonia-de-formigas/bases/Banana/Metricas/metricas_class_' + self.name + '_banana_Reduzido_paralelo.csv')


    # Generate CSV with metrics
    dataCSV = [
        ['', 'Acuracia', 'F-Measure', 'Precisao', 'Recall'],
        ['Agregado', round(accuracy, 2), round(f1_aggregate, 2), round(precision_aggregate, 2), round(recall_aggregate, 2)],
        ['Media', round(results['test_accuracy'].mean(), 4), round(results['test_f1_macro'].mean(), 4), round(results['test_precision_macro'].mean(), 4), round(results['test_recall_macro'].mean(), 4)],
        ['Desvio Padrao', round(results['test_accuracy'].std(), 4), round(results['test_f1_macro'].std(), 4), round(results['test_precision_macro'].std(), 4), round(results['test_recall_macro'].std(), 4)]
    ]

    self.create_csv (dataCSV, 'C:/Users/maria.oliveira/Documents/workspace/projetos/IC-TCC-colonia-de-formigas/bases/Banana/Metricas/metricas_global_' + self.name + '_banana_Reduzido_paralelo.csv')