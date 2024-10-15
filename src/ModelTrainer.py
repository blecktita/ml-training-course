from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class ModelTrainer:
    """Class for training and evaluating logistic regression models."""

    def __init__(self, X, y):
        """
        Initializes the ModelTrainer with data splits.

        Args:
            X (dict): Dictionary containing feature sets for 'train', 'val', and 'test'.
            y (dict): Dictionary containing target variables for 'train', 'val', and 'test'.
        """
        self.X = X
        self.y = y
        self.categorical_features = [
            'job', 'marital', 'education', 'housing', 'contact', 'month', 'poutcome'
        ]
        self.numerical_features = [
            'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'
        ]
        self.model = None
        self.ohe = None

    def prepare_features(self, X_data, fit_ohe=False):
        """
        Prepares feature matrices by encoding categorical variables and combining them with numerical features.

        Args:
            X_data (pd.DataFrame): The dataset to prepare.
            fit_ohe (bool): Whether to fit the OneHotEncoder.

        Returns:
            np.ndarray: The combined feature matrix.
        """
        if fit_ohe:
            self.ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
            X_cat = self.ohe.fit_transform(X_data[self.categorical_features])
        else:
            X_cat = self.ohe.transform(X_data[self.categorical_features])

        X_num = X_data[self.numerical_features].values
        X_combined = np.hstack((X_num, X_cat))
        return X_combined

    def train_model(self, C=1.0, max_iter=1000):
        """
        Trains the logistic regression model.

        Args:
            C (float): Inverse of regularization strength.
            max_iter (int): Maximum number of iterations.

        Returns:
            LogisticRegression: The trained model.
        """
        X_train_combined = self.prepare_features(self.X['train'], fit_ohe=True)
        self.model = LogisticRegression(
            solver='liblinear', C=C, max_iter=max_iter, random_state=42
        )
        self.model.fit(X_train_combined, self.y['train'])
        return self.model

    def evaluate_model(self, dataset='val'):
        """
        Evaluates the model on the specified dataset.

        Args:
            dataset (str): The dataset to evaluate on ('train', 'val', or 'test').

        Returns:
            float: The accuracy score.
        """
        X_data = self.X[dataset]
        y_true = self.y[dataset]
        X_combined = self.prepare_features(X_data)
        y_pred = self.model.predict(X_combined)
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    def feature_elimination(self, features):
        """
        Performs feature elimination to identify the least useful feature.

        Args:
            features (list): List of features to test.

        Returns:
            dict: Dictionary containing accuracy differences for each feature.
        """
        original_accuracy = self.evaluate_model('val')
        differences = {}

        for feature in features:
            # Create copies to avoid modifying original data
            X_train_temp = self.X['train'].drop(columns=[feature])
            X_val_temp = self.X['val'].drop(columns=[feature])

            # Update feature lists
            numerical_temp = [col for col in self.numerical_features if col != feature]
            categorical_temp = [col for col in self.categorical_features if col != feature]

            # Prepare features
            self.categorical_features = categorical_temp
            self.numerical_features = numerical_temp

            X_train_combined = self.prepare_features(X_train_temp, fit_ohe=True)
            X_val_combined = self.prepare_features(X_val_temp)

            # Retrain the model
            self.model.fit(X_train_combined, self.y['train'])

            # Evaluate the model
            y_pred = self.model.predict(X_val_combined)
            acc = accuracy_score(self.y['val'], y_pred)
            diff = original_accuracy - acc
            differences[feature] = diff

            # Reset features for the next iteration
            self.categorical_features = [
                'job', 'marital', 'education', 'housing', 'contact', 'month', 'poutcome'
            ]
            self.numerical_features = [
                'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'
            ]

        return differences

    def hyperparameter_tuning(self, C_values):
        """
        Performs hyperparameter tuning to find the best value of C.

        Args:
            C_values (list): List of C values to test.

        Returns:
            dict: Dictionary containing validation accuracies for each C.
        """
        accuracies = {}

        for C in C_values:
            self.train_model(C=C)
            acc = self.evaluate_model('val')
            accuracies[C] = acc

        return accuracies
