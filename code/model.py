import numpy as np
import tqdm

def gini(*groups):
    """ Gini impurity for classification problems.

    Args: groups (tuple): tuples containing:
        (ndarray): Group inputs (x).
        (ndarray): Group labels (y).

    Returns:
        (float): Gini impurity index.

    """
    m = np.sum([group[0].shape[0] for group in groups])  # Number of samples

    gini = 0.0

    for group in groups:
        y = group[1]
        group_size = y.shape[0]

        # Count number of observations per class
        _, class_count = np.unique(y, return_counts=True)
        proportions = class_count / group_size
        weight = group_size / m

        gini += (1 - np.sum(proportions ** 2)) * weight

    return gini

def entropy(p):
    """
    Entropy of the distribution [p, 1-p]
    https://carbonati.github.io/posts/random-forests-from-scratch/
    """
    if p == 0 or p == 1: return 0
    return - (p * np.log2(p) + (1 - p) * np.log2(1-p))

def information_gain(*groups):
    # TODO
    """Entropy for classification problems.

    Args: groups (tuple): tuples containing:
        (ndarray): Group inputs (x).
        (ndarray): Group labels (y).

    Returns:
        (float): Gini impurity index.
    """
    m = np.sum([group[0].shape[0] for group in groups])  # Number of samples

    entrp = 0.0

    for group in groups:
        y = group[1]
        group_size = y.shape[0]

        # Count number of observations per class
        _, class_count = np.unique(y, return_counts=True)
        proportions = class_count / group_size
        weight = group_size / m

        entrp += np.sum([ entropy(p) for p in proportions]) * weight

    return entrp


def split(x, y, feature_idx, split_value):
    """ Returns two tuples holding two groups resulting from split.

    Args:
        x (ndarray): Input.
        y (ndarray): Labels.
        feature_idx (int): Feature to consider.
        split_value (float): Value used to split.

    Returns:
        (tuple):tuple containing:
            (tuple):tuple containing:
                (ndarray): Inputs of group under split.
                (ndarray): Labels of group under split.
            (tuple):tuple containing:
                (ndarray): Inputs of group over split.
                (ndarray): Labels of group over split.

    """
    bool_mask = x[:, feature_idx] < split_value
    group_1 = (x[bool_mask], y[bool_mask])
    group_2 = (x[bool_mask == 0], y[bool_mask == 0])
    return group_1, group_2


def legal_split(*groups, min_samples_leaf=1):
    """Test if all groups hold enough samples to meet the min_samples_leaf
    requirement """
    for g in groups:
        if g[0].shape[0] < min_samples_leaf:
            return False
    return True


def split_search_feature(x, y, feature_idx, min_samples_leaf, criterion='gini'):
    """Return best split on dataset given a feature.

    Return error values (np.Nan for floats and None for tuples) if no
    split can be found.

    Args:
        x(ndarray): Inputs.
        y(ndarray): Labels.
        feature_idx(int): Index of feature to consider
        min_samples_leaf(int): Minimum number of samples to be deemed
        a leaf node.

    Returns:
        (tuple):tuple containing:
            (float): gini score.
            (float): value used for splitting.
            (tuple):tuple containing:
                (tuple):tuple containing:
                    (ndarray): Inputs of group under split.
                    (ndarray): Labels of group under split.
                (tuple):tuple containing:
                    (ndarray): Inputs of group over split.
                    (ndarray): Labels of group over split.

    """
    scores = []
    splits = []
    split_values = []
    series = x[:, feature_idx]

    # Greedy search on all input values for relevant feature
    for split_value in series:
        s = split(x, y, feature_idx, split_value)

        # Test if groups hold enough samples, otherwise keep searching
        if legal_split(*s, min_samples_leaf=min_samples_leaf):
            scores.append(gini(*s) if criterion=='gini' else information_gain(*s))
            splits.append(s)
            split_values.append(split_value)

    if len(scores) == 0:
        # Impossible to split
        # This will occur when samples are identical in a given node
        return np.NaN, np.NaN, None

    arg_min = np.argmin(scores)

    return scores[arg_min], split_values[arg_min], splits[arg_min]


def split_search(X, Y, min_samples_leaf, feature_search=None,  criterion='gini'):
    """Return best split on dataset.

    Return error values (np.Nan for floats and None for tuples) if no
    split can be found.

    Args:
        x(ndarray): Inputs.
        y(ndarray): Labels.
        feature_search(int): Number of features to use for split search
        min_samples_leaf(int): Minimum number of samples to be deemed
        a leaf node.

    Returns:
        (tuple):tuple containing:
            (int): Index of best feature.
            (float): value used for splitting.
            (tuple):tuple containing:
                (ndarray): Inputs of group under split.
                (ndarray): Labels of group under split.
            (tuple):tuple containing:
                (ndarray): Inputs of group over split.
                (ndarray): Labels of group over split.

    """
    all_scores = []
    splits = []
    split_values = []

    # Flag to handle cases where no legal splits can be found
    split_flag = False

    n_features = X.shape[1]
    if feature_search is None:
        # Default to all features
        feature_indices = np.arange(n_features)
    else:
        if feature_search > n_features:
            raise Exception('Tried searching more features than available features in dataset.')

        # Randomly choose feature_search features to look up
        feature_indices = np.random.choice(n_features, feature_search, replace=False)

    # Search over features
    for feature_idx in feature_indices:
        score, s_value, s = split_search_feature(X, Y, feature_idx, min_samples_leaf, criterion)
        all_scores.append(score)
        split_values.append(s_value)
        splits.append(s)

        if score is not np.NaN:
            # At least one legal split
            split_flag = True

    if not split_flag:
        # Impossible to split
        # This will occur when samples are identical in a given node
        return np.NaN, np.NaN, None, None

    arg_min = np.nanargmin(all_scores)

    group_1, group_2 = splits[arg_min]

    return feature_indices[arg_min], split_values[arg_min], group_1, group_2

"""## Nodes"""

class Node:
    def __init__(self, depth=0):
        """Node definition.

        Args:
            depth(int): Depth of this node (root node depth should be 0).
        """
        self._feature_idx = None  # Feature index to use for splitting
        self._split_value = None
        self._leaf = False
        self._label = None
        self._left_child = None
        self._right_child = None
        self._depth = depth

    def train(self, X, Y,
              feature_search,
              max_depth=8,
              min_samples_split=2,
              min_samples_leaf=1,
              criterion='gini',
              max_leaf_nodes=None, # TODO
              min_impurity_decrease=0.0, # TODO
              verbose=0, # TODO
              ):
        """Training procedure for node.

        Args:
            * X, ndarray(n_samples, n_features) : Inputs
            * Y, ndarray(n_samples) : Labels
            * feature_search, int : Number of features to search during split search.
            * max_depth, int, default=None : The maximum depth of the tree.
            * min_samples_split, int or float, default=2 : The minimum number of samples required to split an internal node
              * If int, then consider min_samples_split as the minimum number.
              * If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
            * min_samples_leaf, int or float, default=1 : The minimum number of samples required to be at a leaf node.
              A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches.
              This may have the effect of smoothing the model, especially in regression.
              * If int, then consider min_samples_leaf as the minimum number.
              * If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
            * criterion, {“gini”, “entropy”, “log_loss”}, default=”gini” : The function to measure the quality of a split.
              Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain
            * max_leaf_nodes, int, default=None : Grow trees with max_leaf_nodes in best-first fashion.
              Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
            * min_impurity_decrease, float, default=0.0
              A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
              The weighted impurity decrease equation is the following:
                N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)
              where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child,
              and N_t_R is the number of samples in the right child.
              N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.
            * verbose, int, default=0 : Controls the verbosity when fitting and predicting.
        """
        if feature_search is None : feature_search = X.shape[1]

        if self._depth < max_depth and X.shape[0] > min_samples_split:

            # Retrieve best split coordinates based on gini impurity and two groups
            self._feature_idx, self._split_value, group_1, group_2 = split_search(X, Y, min_samples_leaf, feature_search, criterion)

            if self._feature_idx is not np.NaN:
                # Recursively split and train child nodes
                self._left_child = Node(self._depth + 1)
                self._right_child = Node(self._depth + 1)
                self._left_child.train(*group_1, feature_search, max_depth, min_samples_split,
                                       min_samples_leaf, criterion, max_leaf_nodes, min_impurity_decrease, verbose)
                self._right_child.train(*group_2, feature_search, max_depth, min_samples_split,
                                        min_samples_leaf, criterion, max_leaf_nodes, min_impurity_decrease, verbose)
            else:
                # Impossible to split. Convert to leaf node
                # This will occur when observations are identical in a given node
                self._sprout(Y)
        else:
            # End condition met. Convert to leaf node
            self._sprout(Y)

    def _sprout(self, Y):
        """Flag node as a leaf node."""
        self._leaf = True

        # Count classes in current node to determine class
        _classes, counts = np.unique(Y, return_counts=True)
        self._label = _classes[np.argmax(counts)]

    def eval(self, X, Y):
        """Return number of correct predictions over a dataset."""
        if self._leaf:
            return np.sum(Y == self._label)
        else:
            group_1, group_2 = split(X, Y, self._feature_idx, self._split_value)
            return self._left_child.eval(*group_1) + self._right_child.eval(*group_2)

    def count(self):
        """Recursively count nodes."""
        if self._leaf: return 1
        return 1 + self._left_child.count() + self._right_child.count()

    def predict(self, x):
        """Recursively predict class for a single individual.

        Args:
            x(ndarray): A single individual.

        Returns:
            (int): Class index.
        """
        if self._leaf:
            return self._label
        else:
            if x[self._feature_idx] < self._split_value:
                return self._left_child.predict(x)
            else:
                return self._right_child.predict(x)

"""## Decision Tree"""

class Tree:
    """Decision tree for classification"""
    def __init__(self,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 feature_search = None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 bootstrap=True,
                 verbose=0,
                 max_samples=None,
                 ):
        """Decision tree for classification.

        Args :
            * criterion, {“gini”, “entropy”, “log_loss”}, default=”gini” : The function to measure the quality of a split.
              Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain,
            * max_depth, int, default=None : The maximum depth of the tree.
            * min_samples_split, int or float, default=2 : The minimum number of samples required to split an internal node
              * If int, then consider min_samples_split as the minimum number.
              * If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
            * min_samples_leaf, int or float, default=1 : The minimum number of samples required to be at a leaf node.
              A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches.
              This may have the effect of smoothing the model, especially in regression.
              * If int, then consider min_samples_leaf as the minimum number.
              * If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
            * feature_search, int : The number of features to consider when looking for the best split Number of features to search during split search.
            * max_leaf_nodes, int, default=None : Grow trees with max_leaf_nodes in best-first fashion.
              Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
            * min_impurity_decrease, float, default=0.0
              A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
              The weighted impurity decrease equation is the following:
                N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)
              where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child,
              and N_t_R is the number of samples in the right child.
              N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.
            * bootstrap, bool, default=True : Whether bootstrap samples (resample dataset with replacement) are used when building trees.
              If False, the whole dataset is used to build each tree.
            * verbose, int, default=0 : Controls the verbosity when fitting and predicting.
            * max_samples, int or float, default=None : If bootstrap is True, the number of samples to draw from X to train each base estimator.
              * If None (default), then draw X.shape[0] samples.
              * If int, then draw max_samples samples.
              * If float, then draw max(round(n_samples * max_samples), 1) samples. Thus, max_samples should be in the interval (0.0, 1.0].
        """
        assert criterion in ["gini", "entropy", "log_loss"]

        self._criterion = criterion
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._feature_search = feature_search
        self._max_leaf_nodes=max_leaf_nodes
        self._min_impurity_decrease = min_impurity_decrease
        self._bootstrap = bootstrap
        self._verbose=verbose
        self._max_samples=max_samples

        # Root node
        self._root = Node()

    def train(self, X, Y):
        """Training routine for tree.

        Args:
            X(ndarray): Inputs (n_samples, n_features)
            Y(ndarray): Labels (n_samples)

        Returns:
            None

        """

        n_samples = X.shape[0]
        if self._bootstrap:
            # Resample with replacement
            if self._max_samples is None : max_samples = n_samples
            elif type(self._max_samples) == int : max_samples = self._max_samples
            elif type(self._max_samples) == float : max_samples = max(round(n_samples * self._max_samples), 1)
            else : raise ValueError("max_samples must be int or float")
            bootstrap_indices = np.random.randint(0, max_samples, max_samples)
            X, Y = X[bootstrap_indices], Y[bootstrap_indices]

        if type(self._min_samples_split) == int : min_samples_split = self._min_samples_split
        elif type(self._min_samples_split) == float : min_samples_split = np.ceil(self._min_samples_split * n_samples)
        else : raise ValueError("min_samples_split must be int or float")

        if type(self._min_samples_leaf) == int : min_samples_leaf = self._min_samples_leaf
        elif type(self._min_samples_leaf) == float : min_samples_leaf = np.ceil(self._min_samples_leaf * n_samples)
        else : raise ValueError("min_samples_leaf must be int or float")

        self._root.train(X, Y,
                         self._feature_search,
                         self._max_depth,
                         min_samples_split,
                         min_samples_leaf,
                         self._criterion,
                         self._max_leaf_nodes,
                         self._min_impurity_decrease,
                         self._verbose
                        )

    def eval(self, X, Y):
        """Return error on dataset"""
        return 100 * (1 - self._root.eval(X, Y) / X.shape[0])

    def node_count(self):
        """Count nodes in tree."""
        return self._root.count()

    def predict(self, x):
        """Predict class for one observation.

        Args:
            x(ndarray): A single observation.

        Returns:
            (int): Predicted class index.

        """
        return self._root.predict(x)

"""## Random forest"""

class RandomForestClassifier:
    """
    Random Forest implementation using numpy
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # https://github.com/sachaMorin/np-random-forest
    # https://carbonati.github.io/posts/random-forests-from-scratch/
    """
    def __init__(self,
                 n_estimators=100,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features='sqrt',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 bootstrap=True,
                 random_state=None,
                 verbose=0,
                 max_samples=None,
          ):
        """Random Forest implementation using numpy.

        Args :
            * n_estimators, int, default=100 : The number of trees in the forest.
            * criterion, {“gini”, “entropy”, “log_loss”}, default=”gini” : The function to measure the quality of a split.
              Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain,
            * max_depth, int, default=None : The maximum depth of the tree.
              If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
            * min_samples_split, int or float, default=2 : The minimum number of samples required to split an internal node
              * If int, then consider min_samples_split as the minimum number.
              * If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
            * min_samples_leaf, int or float, default=1 : The minimum number of samples required to be at a leaf node.
              A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches.
              This may have the effect of smoothing the model, especially in regression.
              * If int, then consider min_samples_leaf as the minimum number.
              * If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
            * max_features, {“sqrt”, “log2”, None}, int or float, default=”sqrt” : The number of features to consider when looking for the best split
              Number of features to search when splitting.
              * If int, then consider max_features features at each split.
              * If float, then max_features is a fraction and max(1, int(max_features * n_features_in_)) features are considered at each split.
              * If “sqrt”, then max_features=floor(sqrt(n_features)).
              * If “log2”, then max_features=floor(log2(n_features)).
              * If None, then max_features=n_features.
            * max_leaf_nodes, int, default=None : Grow trees with max_leaf_nodes in best-first fashion.
              Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
            * min_impurity_decrease, float, default=0.0
              A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
              The weighted impurity decrease equation is the following:
                N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)
              where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child,
              and N_t_R is the number of samples in the right child.
              N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.
            * bootstrap, bool, default=True : Whether bootstrap samples (resample dataset with replacement) are used when building trees.
              If False, the whole dataset is used to build each tree.
            * random_state, int, RandomState instance or None, default=None : Controls both the randomness of the bootstrapping of the samples used
              when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each
              node (if max_features < n_features). See Glossary for details.
            * verbose, int, default=0 : Controls the verbosity when fitting and predicting.
            * max_samples, int or float, default=None : If bootstrap is True, the number of samples to draw from X to train each base estimator.
              * If None (default), then draw X.shape[0] samples.
              * If int, then draw max_samples samples.
              * If float, then draw max(round(n_samples * max_samples), 1) samples. Thus, max_samples should be in the interval (0.0, 1.0].
        """
        assert criterion in ["gini", "entropy", "log_loss"]

        self._no_trees = n_estimators
        self._criterion = criterion
        self._max_depth = max_depth if max_depth is not None else 1e8
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._max_features = max_features
        self._max_leaf_nodes=max_leaf_nodes
        self._min_impurity_decrease = min_impurity_decrease
        # Do not bootstrap if only one tree is requested
        self._bootstrap = bootstrap if self._no_trees > 1 else False
        self._random_state=random_state
        self._verbose=verbose
        self._max_samples=max_samples

        self._trees = []

    def train(self, X, Y):
        """Training procedure.

        Args:
            X(ndarray): Inputs (n_samples, n_features)
            Y(ndarray): Labels (n_samples)

        Returns:
            None
        """
        if self._random_state : np.random.seed(self._random_state)

        max_features = self._max_features if self._no_trees  > 1 else X.shape[1]
        n_features = X.shape[1]
        #  Number of features to search during split search.
        if max_features == "sqrt" : feature_search = int(np.sqrt(n_features))
        elif max_features == "log2" : feature_search = int(np.log2(n_features))
        elif max_features is None : feature_search = n_features
        elif type(max_features) == int : feature_search = max_features
        elif type(max_features) == float : feature_search = max(1, int(max_features * n_features))
        else : raise ValueError("max_features must be {“sqrt”, “log2”, None}, int or float")

        for i in tqdm.tqdm(range(self._no_trees), desc="Training Forest..."):
            tree = Tree(
                 criterion=self._criterion,
                 max_depth=self._max_depth,
                 min_samples_split=self._min_samples_split,
                 min_samples_leaf=self._min_samples_leaf,
                 feature_search=feature_search,
                 max_leaf_nodes=self._max_leaf_nodes,
                 min_impurity_decrease=self._min_impurity_decrease,
                 bootstrap=self._bootstrap,
                 verbose=self._verbose,
                 max_samples=self._max_samples
            )
            tree.train(X, Y)
            self._trees.append(tree)

    def eval(self, X, Y):
        """"Evaluate accuracy on dataset."""
        Y_hat = self.predict(X)
        return np.sum(Y_hat == Y) / X.shape[0]

    def predict(self, X):
        """Return predicted labels for given inputs."""
        return np.array([self._aggregate(X[i]) for i in range(X.shape[0])])

    def _aggregate(self, X):
        """Predict class by pooling predictions from all trees.

        Args:
            X(ndarray): A single example.

        Returns:
            (int): Predicted class index.

        """
        temp = [t.predict(X) for t in self._trees]
        _classes, counts = np.unique(np.array(temp), return_counts=True)

        # Return class with max count
        return _classes[np.argmax(counts)]

    def node_count(self):
        """Return number of nodes in forest."""
        return np.sum([t.node_count() for t in self._trees])
