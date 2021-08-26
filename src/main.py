import scikitplot as skplt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import imblearn
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from numpy.random import seed
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# check version number
# import matplotlib
# from keras import optimizers
# import optuna
print(imblearn.__version__)


"""Os datasets estão armazenados no Google Drive"""

# from google.colab import drive
# drive.mount('/content/drive')

testando = False

datasets_dir = "..\projeto_final_neurais\data"
train_set_path = None
validation_set_path = None
test_set_path = None
# Verificar nome do arquivo de treino. "É training_data", não "traning_data"
if testando:
    train_set_path = f"{datasets_dir}/tiny-data.csv"
    validation_set_path = f"{datasets_dir}/tiny-data.csv"
    test_set_path = f"{datasets_dir}/tiny-data.csv"
else:
    train_set_path = f"{datasets_dir}/training_data.csv"
    validation_set_path = f"{datasets_dir}/validation_data.csv"
    test_set_path = f"{datasets_dir}/test_data.csv"

# Read CSVs and shuffle them
train_df = pd.read_csv(train_set_path)
train_df = train_df.sample(frac=1).reset_index(drop=True)

validation_df = pd.read_csv(validation_set_path)
validation_df = validation_df.sample(frac=1).reset_index(drop=True)

test_df = pd.read_csv(test_set_path)
test_df = test_df.sample(frac=1).reset_index(drop=True)

columns_missing_from_test = list(set(train_df.columns) - set(test_df.columns))
columns_missing_from_test


def get_XY_datasets(df):
    Y = np.array(df.IND_BOM_1_1.tolist())
    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(
        df.drop(['IND_BOM_1_1', 'IND_BOM_1_2'], axis=1).to_numpy())
    return X, Y


train_df = train_df.drop(columns_missing_from_test, axis=1)

train_df.head()

train_df.describe()

print('dataframe shapes')
print(train_df.shape, validation_df.shape, test_df.shape)

# train_df.hist("IND_BOM_1_1")
# validation_df.hist("IND_BOM_1_1")
# test_df.hist("IND_BOM_1_1")


# def under_sample(df):
#     min_count = df[df["IND_BOM_1_1"] == 0].shape[0]
#     under_sample = df[df["IND_BOM_1_1"] == 1].sample(
#         frac=min_count / df[df["IND_BOM_1_1"] == 1].shape[0])
#     return pd.concat([df[df["IND_BOM_1_1"] == 0], under_sample])


# def under_over_sample(df):
#     min_class = df[df["IND_BOM_1_1"] == 0]
#     maj_class = df[df["IND_BOM_1_1"] == 1]
#     discrepancy = maj_class.shape[0] - min_class.shape[0]
#     under_sampled = maj_class.sample(
#         frac=(maj_class.shape[0]-discrepancy/2)/maj_class.shape[0])
#     over_sampled = pd.concat(
#         [min_class]*(int((min_class.shape[0] + discrepancy/2)/min_class.shape[0])+1))
#     return pd.concat([under_sampled, over_sampled], ignore_index=True, sort=False, verify_integrity=False)

# from imblearn.under_sampling import RandomUnderSampler
# undersample = RandomUnderSampler(sampling_strategy='majority')
# X, y = get_XY_datasets(train_df)
# X_resampled, y_resampled = undersample.fit_resample(X, y)

# from imblearn.over_sampling import RandomOverSampler
# oversample = RandomOverSampler(sampling_strategy='not majority')
# X, y = get_XY_datasets(train_df)
# X_resampled, y_resampled = oversample.fit_resample(X, y)

# X.shape

# X_resampled.shape

# len(y_resampled)

# under_over_sample(train_df).hist("IND_BOM_1_1")

# corr_matrix = train_df.corr()

# for col in corr_matrix.columns:
#     print(f"======={col}=======")
#     print(corr_matrix[col][abs(corr_matrix[col]) > 0.5])
#     print()

# print(corr_matrix["IND_BOM_1_1"][abs(corr_matrix['IND_BOM_1_1']) > 0.1])


corr_matrix = None

X_train, Y_train = get_XY_datasets(train_df)
X_validation, Y_validation = get_XY_datasets(validation_df)
X_test, Y_test = get_XY_datasets(test_df)

X_train.shape, X_validation.shape, X_test.shape

"""# Construindo um modelo MLP iterativamente"""

MAX_EPOCHS = 40


print('avail gpus')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.random.set_seed(1)

seed(2)


def MLPModel(n_units=[100], activation_fn="relu", output_activation_fn="sigmoid", loss="mean_squared_error", optimizer="sgd"):
    model = keras.Sequential()
    # , kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    print('X_train.shape: ', X_train.shape)
    model.add(layers.Dense(n_units[0], input_dim=X_train.shape[1],
              activation=activation_fn, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(layers.Dense(12, input_dim=X_train.shape[1],
              activation=activation_fn, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    # for h_n_units in n_units[1:]:
    #     print(f'h_n_units: {h_n_units}')
    #     model.add(layers.Dense(h_n_units, input_dim=X_train.shape[1], activation=activation_fn, kernel_regularizer=tf.keras.regularizers.l2(
    #         0.01), activity_regularizer=tf.keras.regularizers.l2(0.01)))  # , kernel_regularizer=tf.keras.regularizers.l2(0.01)))

    # sgd = optimizers.SGD(lr=0.3, decay=1e-9, momentum=0.015, nesterov=True)
    model.add(layers.Dense(1, activation=output_activation_fn))  # Output
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


# TODO fazer os paranaue
def MLPKeras(n_units=[100], activation_fn="relu", output_activation_fn="sigmoid", loss="mean_squared_error", optimizer="sgd"):
    model = keras.Sequential(
        [
            layers.Input(shape=X_train.shape[1]),
            layers.Dense(n_units[0], input_dim),
            layers.Dense(12, input_dim=X_train.shape[1],
                         activation=activation_fn, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        ]
    )
    return model


model = MLPModel()

early_stopping = EarlyStopping(
    monitor="val_loss", patience=50, min_delta=0.001, restore_best_weights=True)

model.fit(X_train, Y_train,
          batch_size=60,
          epochs=MAX_EPOCHS,
          validation_data=(X_validation, Y_validation),
          callbacks=[early_stopping]
          )

test_loss, test_acc = model.evaluate(
    X_test, Y_test, batch_size=X_test.shape[0]//50)
print(test_loss, test_acc)

# tf.keras.backend.clear_session()

# mlp_pred_class = model.predict(X_validation)
# mlp_pred_scores = model.predict_proba(X_validation)

# ## Printando a matriz de confusão

y_pred = np.array([1 if x > 0.5 else 0 for x in model.predict(X_test)])
print(confusion_matrix(Y_test, y_pred))
sn.heatmap(confusion_matrix(Y_test, y_pred, normalize='true'), annot=True)
print(classification_report(Y_test, y_pred))
plt.show()

# """## Usando a função tangente hiperbólica na ativação"""

# tanh_Y_train = np.where(Y_train == 0, -1, Y_train)
# tanh_Y_validation = np.where(Y_validation == 0, -1, Y_validation)
# tanh_Y_test = np.where(Y_test == 0, -1, Y_test)

# """# Modelo Random Forest"""

X_validation, y_validation = get_XY_datasets(validation_df)

X_test, y_test = get_XY_datasets(test_df)


def objective(trial):
    rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 500)
    rf_max_depth = trial.suggest_int("rf_max_depth", 1, 40, log=True)
    print("Running for {0} estimators and {1} of depth".format(
        rf_n_estimators, rf_max_depth))
    classifier_obj = RandomForestClassifier(
        max_depth=rf_max_depth, n_estimators=rf_n_estimators
    )

    for step in range(100):
        classifier_obj.fit(X_resampled, y_resampled)

        # Report intermediate objective value.
        intermediate_value = classifier_obj.score(X_validation, y_validation)
        trial.report(intermediate_value, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
        return intermediate_value


# classifier_obj = RandomForestClassifier(
#     max_depth=rf_max_depth, n_estimators=rf_n_estimators
# )

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100)

# study.best_params

# """## Tunning 2"""


# def objective_acc(trial):
#     rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 500)
#     rf_max_depth = trial.suggest_int("rf_max_depth", 1, 40, log=True)
#     print("Running for {0} estimators and {1} of depth".format(
#         rf_n_estimators, rf_max_depth))
#     classifier_obj = RandomForestClassifier(
#         max_depth=rf_max_depth, n_estimators=rf_n_estimators
#     )

#     for step in range(100):
#         classifier_obj.fit(X_resampled, y_resampled)
#         _y_pred = classifier_obj.predict(X_validation)

#         # Report intermediate objective value.
#         intermediate_value = accuracy_score(y_validation, _y_pred)
#         trial.report(intermediate_value, step)

#         # Handle pruning based on the intermediate value.
#         if trial.should_prune():
#             raise optuna.TrialPruned()
#         return intermediate_value


# study2 = optuna.create_study(direction="maximize")
# study2.optimize(objective_acc, n_trials=50)

# clf = RandomForestClassifier(max_depth=20, random_state=0, min_samples_leaf=1)

# clf.fit(X_resampled, y_resampled)

# clf.score(X_test, y_test)


# def compute_performance_metrics(y, y_pred_class, y_pred_scores=None):
#     accuracy = accuracy_score(y, y_pred_class)
#     recall = recall_score(y, y_pred_class, average='micro')
#     precision = precision_score(y, y_pred_class, average='micro')
#     f1 = f1_score(y, y_pred_class, average='micro')
#     performance_metrics = (accuracy, recall, precision, f1)
#     if y_pred_scores is not None:
#         skplt.metrics.plot_ks_statistic(y, y_pred_scores)
#         plt.show()
#         y_pred_scores = y_pred_scores[:, 1]
#         auroc = roc_auc_score(y, y_pred_scores)
#         aupr = average_precision_score(y, y_pred_scores)
#         performance_metrics = performance_metrics + (auroc, aupr)
#     return performance_metrics


# def print_metrics_summary(accuracy, recall, precision, f1, auroc=None, aupr=None):
#     print()
#     print("{metric:<18}{value:.4f}".format(metric="Accuracy:", value=accuracy))
#     print("{metric:<18}{value:.4f}".format(metric="Recall:", value=recall))
#     print("{metric:<18}{value:.4f}".format(
#         metric="Precision:", value=precision))
#     print("{metric:<18}{value:.4f}".format(metric="F1:", value=f1))
#     if auroc is not None:
#         print("{metric:<18}{value:.4f}".format(metric="AUROC:", value=auroc))
#     if aupr is not None:
#         print("{metric:<18}{value:.4f}".format(metric="AUPR:", value=aupr))


# pred_class = clf.predict(X_test)
# pred_scores = clf.predict_proba(X_test)
# accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(
#     y_test, pred_class, pred_scores)
# print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)

# clf.predict(X_test)


# y_pred = np.array([1 if x > 0.75 else 0 for x in clf.predict(X_test)])
# print(confusion_matrix(y_test, y_pred))
# sn.heatmap(confusion_matrix(y_test, y_pred, normalize='true'), annot=True)
# print(classification_report(y_test, y_pred))
