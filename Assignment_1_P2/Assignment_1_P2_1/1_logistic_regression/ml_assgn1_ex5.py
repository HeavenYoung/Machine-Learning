from load_data_ex1 import *
from normalize_features import *
from gradient_descent_training import *
from return_test_set import *
import matplotlib.pyplot as plt
import os

figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# This loads our data
X, y = load_data_ex1()

# Create the features x1*x2, x1^2, x2^2, x1^3 and x2^3
#########################################
# Write your code here
# Compute the new features
# Insert extra singleton dimension, to obtain Nx1 shape
# Append columns of the new features to the dataset, to the dimension of columns (i.e., 1)
########################################/

rows = X.shape[0]
feature1_array = np.array([])
feature2_array = np.array([])
feature3_array = np.array([])
feature4_array = np.array([])
feature5_array = np.array([])

for i in range(rows):
    feature1_array = np.append(feature1_array, X[i, 0] * X[i, 1])
    feature2_array = np.append(feature2_array, X[i, 0] ** 2)
    feature3_array = np.append(feature3_array, X[i, 1] ** 2)
    # feature4_array = np.append(feature4_array, X[i, 0] ** 3)
    # feature5_array = np.append(feature5_array, X[i, 1] ** 3)

X = np.insert(X, 2, feature1_array, axis=1)
X = np.insert(X, 3, feature2_array, axis=1)
X = np.insert(X, 4, feature3_array, axis=1)
# X = np.insert(X, 5, feature4_array, axis=1)
# X = np.insert(X, 6, feature5_array, axis=1)

# split the dataset into training and test set, using random shuffling
train_samples = 60
X_train, y_train, X_test, y_test = return_test_set(X, y, train_samples)

# Compute mean and std on train set
# Normalize both train and test set using these mean and std values
X_train_normalized, mean_vec, std_vec = normalize_features(X_train)
X_test_normalized = normalize_features(X_test, mean_vec, std_vec)

# After normalizing, we append a column of ones to X_normalized, as the bias term
# We append the column to the dimension of columns (i.e., 1)
# We do this for both train and test set
column_of_ones = np.ones((X_train_normalized.shape[0], 1))
X_train_normalized = np.append(column_of_ones, X_train_normalized, axis=1)
column_of_ones = np.ones((X_test_normalized.shape[0], 1))
X_test_normalized = np.append(column_of_ones, X_test_normalized, axis=1)

# Initialise trainable parameters theta
#########################################
# Write your code here
########################################/

theta = np.zeros((6))
# Set learning rate alpha and number of iterations
alpha = 0.01
iterations = 100

# Call the gradient descent function to obtain the trained parameters theta_final
theta_final, cost_vector_train, cost_vector_test = gradient_descent_training(X_train_normalized, y_train, X_test_normalized, y_test, theta, alpha, iterations)

min_train_cost = np.min(cost_vector_train)
argmin_train_cost = np.argmin(cost_vector_train)
min_test_cost = np.min(cost_vector_test)
argmin_test_cost = np.argmin(cost_vector_test)
print('Final train cost: {:.5f}'.format(cost_vector_train[-1]))
print('Minimum train cost: {:.5f}, on iteration #{}'.format(min_train_cost, argmin_train_cost+1))
print('Final test cost: {:.5f}'.format(cost_vector_test[-1]))
print('Minimum test cost: {:.5f}, on iteration #{}'.format(min_test_cost, argmin_test_cost+1))

# Plot the cost for all iterations
fig, ax1 = plt.subplots()
plot_cost_train_test(cost_vector_train, cost_vector_test, ax1)
plot_filename = os.path.join(os.getcwd(), 'figures', 'ex5_train_test_cost.png')
plt.savefig(plot_filename)

# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()