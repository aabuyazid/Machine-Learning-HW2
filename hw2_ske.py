import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

'''
General Instructions:

1. Do not use any additional libraries. Your code will be tested in a pre-built environment with only 
the library above available.

2. You are expected to fill in the skeleton code precisely as per provided. On top of skeleton code given,
you may write whatever deemed necessary to complete the assignment. For example, you may define additional 
default arguments, class parameters, or methods to help you complete the assignment.

3. Some initial steps or definition are given, aiming to help you getting started. As long as you follow 
the argument and return type, you are free to change them as you see fit.

4. Your code should be free of compilation errors. Compilation errors will result in 0 marks.
'''

TARGET_VARIABLE = 'y'

# 'unknown' is kept (as the lowest value) so no rows are lost to NaN
EDUCATION_ORDER = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}
MONTH_ORDER = {m: i for i, m in enumerate(
    ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], start=1)}
BINARY_COLUMNS = ['default', 'housing', 'loan']

# Unordered categoricals get one fixed numeric code each (not one-hot), so the feature matrix has exactly
# one column per original feature (16). The code values are arbitrary labels, not a meaningful order.
NOMINAL_CODES = {
    'job': {j: i for i, j in enumerate(
        ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired',
         'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])},
    'marital': {'single': 0, 'married': 1, 'divorced': 2},
    'contact': {'unknown': 0, 'cellular': 1, 'telephone': 2},
    'poutcome': {'unknown': 0, 'failure': 1, 'other': 2, 'success': 3},
}

'''
Problem A-1: Data Preprocessing and EDA
'''
class DataLoader:
    '''
    This class will be used to load the data and perform initial data processing. Fill in functions.
    You are allowed to add any additional functions which you may need to check the data. This class 
    will be tested on the pre-built enviornment with only numpy and pandas available.
    '''

    def __init__(self, data_root: str, random_state: int = 42, split: float = 0.8):
        '''
        Inialize the DataLoader class with the data_root path.
        Load data as pd.DataFrame, store as needed and initialize other variables.
        All dataset should save as pd.DataFrame.
        '''
        self.random_state = random_state
        self.data_root = data_root
        self.split = split
        np.random.seed(self.random_state)

        self.data = pd.read_csv(data_root, delimiter=';')

        self.data_train = None
        self.data_valid = None

        self.is_data_prepped = False
        self.is_data_splitted = False

        self.data_prep()
        self.data_split()

    def data_split(self) -> None:
        '''
        You are asked to split the training data into train/valid datasets on the ratio of 80/20. 
        Add the split datasets to self.data_train, self.data_valid. Both of the split should still be pd.DataFrame.
        '''
        if self.is_data_splitted:
            return
        
        # int() floors (3616 of 4521); sample(frac=...) would round to 3617
        n_train = int(self.split * len(self.data))
        self.data_train = self.data.sample(n=n_train, random_state=self.random_state)
        self.data_valid = self.data.drop(self.data_train.index)

        self.is_data_splitted = True

    def data_prep(self) -> None:
        '''
        You are asked to drop any rows with missing values and map categorical variables to numeric values. 
        '''
        if self.is_data_prepped:
            return

        self.data[TARGET_VARIABLE] = self.data[TARGET_VARIABLE].map(lambda x: 1 if x == 'yes' else 0)
        self.data['education'] = self.data['education'].map(EDUCATION_ORDER)  # pyright: ignore[reportArgumentType]
        self.data['month'] = self.data['month'].map(MONTH_ORDER)  # pyright: ignore[reportArgumentType]
        for col in BINARY_COLUMNS:
            self.data[col] = self.data[col].map({'no': 0, 'yes': 1})  # pyright: ignore[reportArgumentType]
        for col, codes in NOMINAL_CODES.items():
            self.data[col] = self.data[col].map(codes)  # pyright: ignore[reportArgumentType]

        # A map() that hits an unexpected value yields NaN, which makes that column float64 even after
        # dropna removes the row. Cast every column to one explicit dtype (np.int64, not the platform-
        # dependent builtin int) so the result doesn't depend on pandas/numpy version or OS.
        self.data = self.data.dropna()
        self.data = self.data.astype(np.int64)

        self.is_data_prepped = True

    def extract_features_and_label(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        '''
        This function will be called multiple times to extract features and labels from train/valid/test 
        data.
        
        Expected return:
            X_data: np.ndarray of shape (n_samples, n_features) - Extracted features
            y_data: np.ndarray of shape (n_samples,) - Extracted labels
        '''
        # explicit dtypes: one dtype for all features, integer labels (np.bincount needs ints)
        y_data = data[TARGET_VARIABLE].to_numpy(dtype=np.int64)
        X_data = data.drop(columns=[TARGET_VARIABLE]).to_numpy(dtype=np.float64)
        
        return (X_data, y_data)


'''
Porblem A-2: Classification Tree Inplementation
'''
class ClassificationTree:
    '''
    You are asked to implement a simple classification tree from scratch. This class will be tested on the
    pre-built enviornment with only numpy and pandas available.

    You may add more variables and functions to this class as you see fit.
    '''
    class Node:
        '''
        A data structure to represent a node in the tree.
        '''
        def __init__(self, split: tuple[int, float] | None = None, left=None, right=None, prediction=None):
            '''
            split: tuple - (feature_idx, split_value)
            left: Node - Left child node
            right: Node - Right child node
            prediction: (any) - Prediction value if the node is a leaf
            left: <
            right: >= 
            '''
            self.split = split
            self.left = left
            self.right = right
            self.prediction = prediction 

        def is_leaf(self):
            return self.prediction is not None

    def __init__(self, random_state: int, max_depth: int = 5):
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.max_depth = max_depth

        self.tree_root = None

    def split_crit(self, y: np.ndarray) -> float:
        '''
        Implement the impurity measure of your choice here. Return the impurity value.
        '''
        n = y.size
        p = np.unique(y, return_counts=True)[1] / n

        return 1 - np.sum(p * p)

    def build_tree_helper(self, X: np.ndarray, y: np.ndarray, depth: int, parent: Node) -> None:
        '''
        Implement the tree building algorithm here. You can recursivly call this function to build the 
        tree. After building the tree, store the root node in self.tree_root.

        Think about the difference between depth here and the max_depth parameter in the constructor.
        '''
        if depth == self.max_depth or (best_split := self.search_best_split(X, y)) is None:
            prediction = np.bincount(y).argmax()
            parent.prediction = prediction

            return
        
        j, t = best_split 
        column_vals = X[:, j]
        mask = column_vals < t
        X_left, y_left, X_right, y_right = X[mask], y[mask], X[~mask], y[~mask]

        parent.left, parent.right = self.Node(), self.Node()
        parent.split = best_split
        
        self.build_tree_helper(X_left, y_left, depth+1, parent.left)
        return self.build_tree_helper(X_right, y_right, depth+1, parent.right)
        
    def build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> None:
        '''
        Implement the tree building algorithm here. You can recursivly call this function to build the 
        tree. After building the tree, store the root node in self.tree_root.

        Think about the difference between depth here and the max_depth parameter in the constructor.
        '''
        self.tree_root = self.Node()

        self.build_tree_helper(X, y, depth=0, parent=self.tree_root)
            
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Fit the classification tree to the training data. This is the function that will be called to train your model in autograder.
        
        Args:
            X: numpy array of shape (n_samples, n_features) - Training features
            y: numpy array of shape (n_samples,) - Training labels
        '''

        return self.build_tree(X, y, depth=0)

    def search_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float] | None:
        '''
        Implement the search for best split here.

        Expected return:
        - tuple(int, float): Best feature index and split value
        - None: If no split is found
        '''
        n = y.size

        best_split = None
        min_crit = self.split_crit(y)

        for j in range(X.shape[1]):
            column_vals = X[:, j]
            sorted_vals = np.unique(column_vals)
            thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2

            for t in thresholds:
                y_left, y_right = y[column_vals < t], y[column_vals >= t]
                n_left, n_right = len(y_left), len(y_right)

                if n_left == 0 or n_right == 0:
                    continue
                
                curr_crit = ( (n_left / n) * self.split_crit(y_left) + 
                              (n_right / n ) * self.split_crit(y_right) )

                if curr_crit < min_crit - 1e-12:
                    min_crit = curr_crit
                    
                    best_split = (j, t)

        return best_split

    def predict_helper(self, X: np.ndarray, curr: Node) -> int:
        if curr.is_leaf():
            assert curr.prediction is not None, "how did this happen"

            return curr.prediction

        assert curr.split is not None, "how did this happen"

        j, t = curr.split
        
        if X[j] < t:
            return self.predict_helper(X, curr.left) # pyright: ignore[reportArgumentType]
        
        return self.predict_helper(X, curr.right) # pyright: ignore[reportArgumentType]

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict classes for multiple samples.
        
        Args:
            X: numpy array with the same columns as the training data
            
        Returns:
            np.ndarray: Array of predictions
        '''
        
        predictions = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            predictions[i] = self.predict_helper(x, self.tree_root)

        return predictions

def train_XGBoost() -> dict:
    '''
    See instruction for implementation details. This function will be tested on the pre-built enviornment
    with numpy, pandas, xgboost available.
    '''
    ...

'''
Initialize the following variable with the best model you have found. This model will be used in testing 
in our pre-built environment.
'''
my_best_model = XGBClassifier()


if __name__ == "__main__":
    dataloader = DataLoader('hw2-bank_data.csv')
    dataloader.data_split()