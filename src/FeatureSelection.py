from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import numpy as np

class FeatureSelector:
    """Class for feature selection and analysis."""

    def __init__(self, X_train, y_train):
        """
        Initializes the FeatureSelector with training data.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
        """
        self.X_train = X_train.copy()
        self.y_train = y_train
        self.categorical_vars = [
            'job', 'marital', 'education', 'housing', 'contact', 'month', 'poutcome'
        ]

    def mode_of_column(self, column):
        """
        Calculates the mode of a specified column.

        Args:
            column (str): The column name.

        Returns:
            The mode of the column.
        """
        mode_value = self.X_train[column].mode()[0]
        return mode_value

    def calculate_correlation_matrix(self):
        """
        Calculates the correlation matrix for numerical features.

        Returns:
            pd.DataFrame: The correlation matrix.
        """
        numerical_features = self.X_train.select_dtypes(include=[np.number]).columns
        corr_matrix = self.X_train[numerical_features].corr()
        return corr_matrix

    def compute_mutual_info(self):
        """
        Computes mutual information scores for categorical variables.

        Returns:
            pd.Series: Mutual information scores for each categorical variable.
        """
        for col in self.categorical_vars:
            le = LabelEncoder()
            self.X_train[col] = le.fit_transform(self.X_train[col])

        mi_scores = mutual_info_classif(
            self.X_train[self.categorical_vars],
            self.y_train,
            discrete_features=True
        )
        mi_scores = pd.Series(mi_scores, index=self.categorical_vars).sort_values(ascending=False)
        mi_scores_rounded = mi_scores.apply(lambda x: round(x, 2))
        return mi_scores_rounded
