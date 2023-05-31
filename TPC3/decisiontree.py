from dataset import Dataset

import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

""""***Escolha de atributos": entropy, gini index ou gini ratio***

Na função **new_tree**, para dividirmos os dados em ramos da melhor forma é preciso calcular qual é o melhor critério de divisão e, consequentemente, obtemos a melhor feature e o valor do threshold, que nos diz onde se divide (pela função **best_criterion**). Depois de dividirmos entre right e left, cada lado da arvore chama a função recursivamente.

**"Post-prunnig": Pessimistic error ou Reduced error**

O principal objetivo é combater o overfitting. Assim, é calculada a label mais comum da subtree (**most_common_label**) e é usado esse valor de resposta final (label) para podar a subtree em questão.

**"Pre-prunning": Independence, Max Depth ou Size**

O principal objetivo é evitar o overfitting, ou seja, a um certo ponto, impede que a árvore continue a crescer e coloca como resposta final (label) o calculado também por **most_common_label**.
"""

class DecisionTree:

    #method 'init'
    def __init__(self, chosen_criterion = 'entropy', prunning = 'pre', max_depth = None, max_features = None, min_samples = 2, min_samples_leaf = 1,pre_prune = 'size', post_prune = None, pre_pruning_threshold = 8, post_pruning_threshold = 0.01):
        self.chosen_criterion = chosen_criterion 
        self.max_depth = max_depth   #controls the maximum depth of the decision tree, avoids overfitting
        self.max_features = max_features
        self.min_samples = min_samples    #minimum number of samples required to split an internal node
        self.min_samples_leaf = min_samples_leaf    #minimum number of samples required to be at a leaf node.
        self.pre_prune = pre_prune    #size, independence or max_depth
        self.pre_pruning_threshold = pre_pruning_threshold
        self.post_prune = post_prune  #pessimistic_error or reduced_error
        self.post_pruning_threshold = post_pruning_threshold
        self.tree = None
        
    #method 'fit' builds the new tree
    def fit(self, X, y):
        self.tree = self.new_tree(X, y)

    def traverse_tree(self, x, tree):  # traverse the tree to make a prediction
        if tree.label is not None:
            return tree.label
        
        if x[tree.feature] <= tree.threshold:
            return self.traverse_tree(x, tree.left)
        else:
            return self.traverse_tree(x, tree.righ)

    def predict(self, X):
        predictions = [self.traverse_tree(x, self.tree) for x in X]
        return np.array(predictions)

    #auxiliar function used for 'new_tree' stopping criterion
    def is_done(self, X, y, depth):
        min_samples = 2
        num_samples = X.shape[0]
        num_labels = len(np.unique(y))

        if (depth >= self.max_depth
            or num_labels == 1
            or num_samples < self.min_samples
            or num_samples < self.min_samples_leaf):
            return True
        return False

    #recursive function that builds the tree
    def new_tree(self, X, y, depth=0):
        num_features = X.shape[1]

        #if X is empty
        if len(X) == 0:
            return None
        #if there is only one label in the input data
        if len(np.unique(y)) == 1:
            return Node(label=y[0])

        #stopping criterion
        if self.is_done(X, y, depth):
           return Node(label = self.most_common_label(y))


        #----- PRE-PRUNNING ------
        if self.pre_prune == 'independence' and num_features == 1:
            return Node(label = self.most_common_label(y))
        elif self.pre_prune == 'max_depth' and depth >= self.max_depth:
            return Node(label = self.most_common_label(y))
        elif self.pre_prune ==  'size' and len(X) < self.pre_pruning_threshold:
            return Node(label = self.most_common_label(y))


        #the feature and threshold to split the data
        feature, threshold = self.best_criterion(X, y)


        #----- POST-PRUNNING ------
        #if the pessimistic error of the decision tree after splitting is greater than the pessimistic error of the parent node
        if self.post_prune == 'pessimistic_error' and self.check_pessimistic_error(X, y, feature, threshold):
            return Node(label = self.most_common_label(y))
        
        #if the difference between the error of the current node and the error of the subtree rooted at this node is less than or equal to a predefined threshold
        elif self.post_prune == 'reduced_error' and self.check_reduced_error(X, y, feature, threshold):
            return Node(label = self.most_common_label(y))


        #split the data and grow tree recursively
        left, right = self.split(X[:, feature], threshold)
        left_tree = self.new_tree(X[left, :], y[left], depth + 1)
        right_tree = self.new_tree(X[right, :], y[right], depth + 1)
        
        return Node(feature=feature, threshold=threshold, left=left_tree, right=right_tree)   
    
    
    #returns the best feature and threshold to split the data into groups through the best criterion
    def best_criterion(self, X, y):
        num_features = X.shape[1]
        best_impurity = 1
        split_feature, split_threshold = None, None

        if self.chosen_criterion == 'gini_index':
            for feature in range(num_features):
                X_column = X[:, feature]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    #the impurity on the left side of the threshold has already been evaluated in a previous split
                    _, y_right = self.split(X_column, threshold)
                    gini = self.gini_index(y_right)
                    
                    #selects the split with the lowest impurity score
                    if gini < best_impurity:
                        best_impurity = gini
                        split_feature = feature
                        split_threshold = threshold

        elif self.chosen_criterion == 'entropy':
            for feature in range(num_features):
                X_column = X[:, feature]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    _, y_right = self.split(X_column, threshold)
                    entropy = self.entropy(y_right)
                    
                    #selects the split with the lowest impurity score
                    if entropy < best_impurity:
                        best_impurity = entropy
                        split_feature = feature
                        split_threshold = threshold

        elif self.chosen_criterion == 'gain_ratio':
            for feature in range(num_features):
                X_column = X[:, feature]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    _, y_right = self.split(X_column, threshold)
                    gain_ratio = self.gain_ratio(X_column, y_right)
                    
                    #selects the split with the highest impurity score
                    if gain_ratio > best_impurity:
                        best_impurity = gain_ratio
                        split_feature = feature
                        split_threshold = threshold
        
        return split_feature, split_threshold

    #returns the most common label as a result
    def most_common_label(self, y):
        c = Counter(y)
        return c.most_common(1)[0][0]

    #splits a set of data into 'left' and 'right'
    def split(self, X_column, threshold):
        left = np.argwhere(X_column <= threshold).flatten()
        right = np.argwhere(X_column > threshold).flatten()

        return left, right
    


    #---------- all criteria to split the data: entropy, gini index and gini ratio ---------------
    
    #calculates the entropy of a set of data
    def entropy(self, y):
        n_samples = len(y)
        proportions = np.bincount(y) / n_samples
        res = -np.sum([p * np.log2(p) for p in proportions if p > 0])

        return res
    
    #calculates the Gini index of a set of data
    def gini_index(self, y):
        n_samples = len(y)
        counts = np.unique(y, return_counts=True)[1]
        proportion = counts / n_samples
        gini = 1 - np.sum(proportion ** 2)

        return gini
    
    #calculates the Gain ratio of a set of data
    def gain_ratio(self, feature, y):
        n_samples = len(y)
        values, counts = np.unique(feature, return_counts=True)
        entropy_value = self.entropy(y)
        IV = - np.sum((counts / n_samples) * np.log2(counts / n_samples))
        IG = entropy_value

        for value, count in zip(values, counts):
            subset_y = y[feature == value]
            IG -= (count / n_samples) * self.entropy(subset_y)

        return IG / IV if IV != 0 else 0

    

    #---------- Post-Prunning: Pessimistic Error and Reduced Error ---------------

    #calculates a pessimistic error estimate for each leaf node of the new tree
    def pessimistic_error_value(self, y, p):
        return p + 1.96 * np.sqrt((p * (1 - p)) / len(y))

    #calculates the proportion of labels that don't belong to the most common label
    def reduced_error_value(self, y):
        _, counts = np.unique(y, return_counts=True)
        proportion = np.max(counts) / np.sum(counts)
        return 1 - proportion


    def check_pessimistic_error(self, X, y, feature_idx, threshold):
        left = X[:, feature_idx] < threshold
        right = X[:, feature_idx] >= threshold
        
        p = len(left) / len(y)   #proportion of data that belong to the left branch
        error = self.pessimistic_error_value(y, p)  #calculating the error

        return error < self.post_pruning_threshold   #True if the error is less than the threshold


    def check_reduced_error(self, X, y, feature_idx, threshold):
        left = X[:, feature_idx] < threshold
        right = X[:, feature_idx] >= threshold
        
        error_left = self.reduced_error_value(y[left])    #calculating the error
        error_right = self.reduced_error_value(y[right])

        error = (len(y[left]) * error_left + len(y[right]) * error_right) / len(y)
        
        return error < self.post_pruning_threshold    #True if the error is less than the threshold






    # Resolução de conflitos
    def prune(self):
        pass

    def majority_voting(self):
        pass

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Load the dataset and print
data = Dataset.from_csv('/content/iris.csv', label='Species')

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(data.X.transpose(), data.y, test_size = 0.2, random_state = 2023)

#decision tree model and fit method
f = DecisionTree(max_depth=10, chosen_criterion = 'entropy')
f.fit(X_train, y_train)

#predict method and accuracy
y_pred = f.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
    
print("Accuracy:", accuracy)