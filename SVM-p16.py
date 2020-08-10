'''
Name:       Kevin Chen
Professor:  Haim Schweitzer
Due Date:   8/2/2020 
Project 3 - Polynomial 16
Python 3.7.6
'''

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
import numpy as np

# ====================================
# STEP 1: read the training and testing data.
# Do not change any code of this step.

# specify path to training data and testing data
train_x_location = "x_train16.csv"
train_y_location = "y_train16.csv"

test_x_location = "x_test.csv"
test_y_location = "y_test.csv"

print("Reading training data")
x_train = np.loadtxt(train_x_location, dtype="uint8", delimiter=",")
y_train = np.loadtxt(train_y_location, dtype="uint8", delimiter=",")

m, n = x_train.shape  # m training examples, each with n features
m_labels,  = y_train.shape  # m2 examples, each with k labels
l_min = y_train.min()

assert m_labels == m, "x_train and y_train should have same length."
assert l_min == 0, "each label should be in the range 0 - k-1."
k = y_train.max()+1

print(m, "examples,", n, "features,", k, "categories.")

print("Reading testing data")
x_test = np.loadtxt(test_x_location, dtype="uint8", delimiter=",")
y_test = np.loadtxt(test_y_location, dtype="uint8", delimiter=",")

m_test, n_test = x_test.shape
m_test_labels,  = y_test.shape
l_min = y_train.min()

assert m_test_labels == m_test, "x_test and y_test should have same length."
assert n_test == n, "train and x_test should have same number of features."

print(m_test, "test examples.")


# ====================================
# STEP 2: pre processing
# Please modify the code in this step.

print("Pre processing data")

np.nan_to_num(x_train)
np.nan_to_num(x_test)

# Adding normalized noise to the training data
x_train_noise = x_train.copy()
np.random.seed(1)
x_train_noise = x_train_noise + np.random.normal(0, 0.1, x_train.shape)
y_train_noise = y_train.copy()

x_train = np.append(x_train, x_train_noise, axis=0)
y_train = np.append(y_train, y_train_noise, axis=0)

# Scaling the data
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)
x_test = min_max_scaler.transform(x_test)


# ====================================
# STEP 3: train model.
# Please modify the code in this step.

print("---train")

C_range = np.logspace(-2, 10, 10)
gamma_range = np.logspace(-9, 3, 10)
param_grid = dict(
    gamma=gamma_range, C=C_range)
cv = StratifiedKFold(n_splits=10, shuffle=True)
grid = GridSearchCV(SVC(kernel="poly"),
                    param_grid=param_grid, cv=cv, n_jobs=-1, scoring="balanced_accuracy")
grid.fit(x_train, y_train)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

model = SVC(kernel="poly", C=grid.best_params_[
    "C"], gamma=grid.best_params_["gamma"], coef0=1, degree=3, cache_size=2000)

model.fit(x_train, y_train)


# ====================================
# STEP3: evaluate model
# Don"t modify the code below.

print("---evaluate")
print(" number of support vectors: ", model.n_support_)
acc = model.score(x_test, y_test)
#print(acc)
print("acc:", acc)
