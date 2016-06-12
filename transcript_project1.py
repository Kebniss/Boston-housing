import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

# Make matplotlib show our plots inline (nicely formatted in the notebook) only to be used in the browser IDE
#%matplotlib inline

# Create our client's feature set for which he will predicting a selling price
CLIENT_FEATURES = [[11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]]

# Load the Boston Housing dataset into the city_data variable. This dataset is already contained 
# in sklearn and as such it has its own loading command. 
city_data = datasets.load_boston()

# Initialize the housing prices and housing features. target and data are features of the boston dataset
housing_prices = city_data.target
housing_features = city_data.data

print "Boston Housing dataset loaded successfully!"

# # how to have a quick description of the data with numpy
# print city_data.keys(),"\n" # summary of what is available
# print city_data.data[0:1],"\n" # to check out what the data looks like
# print city_data.feature_names,"\n" # this is what you are looking for
# print city_data.DESCR,"\n" # all the other info you need
# print city_data.target[0:1],"\n" # house prices

# # and almost the same with pandas
# df = pd.DataFrame(data = city_data.data, columns = city_data.feature_names)

# df.head()
# df.describe()

# Number of houses in the dataset
total_houses = np.size(housing_features, 0)

# Number of features in the dataset
total_features = np.size(housing_features, 1)

# Minimum housing value in the dataset
minimum_price = np.amin(housing_prices)

# Maximum housing value in the dataset
maximum_price = np.amax(housing_prices)

# Mean house value of the dataset
mean_price = np.mean(housing_prices)

# Median house value of the dataset
median_price = np.median(housing_prices)

# Standard deviation of housing values of the dataset
std_dev = np.std(housing_prices)

# Show the calculated statistics
# print "Boston Housing dataset statistics (in $1000's):\n"
# print "Total number of houses:", total_houses
# print "Total number of features:", total_features
# print "Minimum house price:", minimum_price
# print "Maximum house price:", maximum_price
# print "Mean house price: {0:.3f}".format(mean_price)
# print "Median house price:", median_price
# print "Standard deviation of house price: {0:.3f}".format(std_dev)`

# Put any import statements you need for this code block here
from sklearn import cross_validation

def shuffle_split_data(X, y):
    """ Shuffles and splits data into 70% training and 30% testing subsets,
        then returns the training and testing subsets. """
    rs_X = cross_validation.ShuffleSplit(np.size(X), n_iter = 1, test_size = .3, 
                                       random_state = 0)
    
    # Shuffle and split the data
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    
    rs_y = cross_validation.ShuffleSplit(np.size(y), n_iter = 1, test_size = .3, 
                                       random_state = 0)
    
    for train_index, test_index in rs_X:
        X_train = train_index
        X_test = test_index
        
    for train_index, test_index in rs_y:
        y_train = train_index
        y_test = test_index

    # Return the training and testing data subsets
    return X_train, y_train, X_test, y_test


# Test shuffle_split_data
try:
    X_train, y_train, X_test, y_test = shuffle_split_data(housing_features, housing_prices)
    print "Successfully shuffled and split the data!"
except:
    print "Something went wrong with shuffling and splitting the data."


# Put any import statements you need for this code block here
from sklearn.metrics import mean_squared_error

def performance_metric(y_true, y_predict):
    """ Calculates and returns the total error between true and predicted values
        based on a performance metric chosen by the student. """

    error = mean_squared_error(y_true, y_predict)
    return error


# Test performance_metric
try:
    total_error = performance_metric(y_train, y_train)
    print "Successfully performed a metric calculation!"
except:
    print "Something went wrong with performing a metric calculation."


# Put any import statements you need for this code block
from sklearn.metrics import make_scorer
from sklearn import grid_search

def fit_model(X, y):
    """ Tunes a decision tree regressor model using GridSearchCV on the input data X 
        and target labels y and returns this optimal model. """

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # Set up the parameters we wish to tune
    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}

    # Make an appropriate scoring function
    scoring_function = make_scorer(mean_squared_error, greater_is_better = False)

    # Make the GridSearchCV object. It returns an object that has an attribute 
    # best_estimator (only if refit == true, default) that has the model with the paramenter 
    # that better represent the data (= best tree depth). This passage doesn't calculate 
    # anything it just sets create the object grid search and set it in reg 
    reg = grid_search.GridSearchCV(regressor, parameters, scoring_function)

    # Fit the learner to the data to obtain the optimal model with tuned parameters. 
    # The best model will be saved in reg.best_estimator
    reg.fit(X, y)

    # Return the optimal model
    return reg.best_estimator_


# Test fit_model on entire dataset
try:
    reg = fit_model(housing_features, housing_prices)
    print "Successfully fit a model!"
except:
    print "Something went wrong with fitting a model."

