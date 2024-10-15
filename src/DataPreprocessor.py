from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """Class for preprocessing the Bank Marketing dataset."""

    def __init__(self, df):
        """
        Initializes the DataPreprocessor with the dataset.

        Args:
            df (pd.DataFrame): The dataset to preprocess.
        """
        self.df = df.copy()
        self.features = [
            'age', 'job', 'marital', 'education', 'balance', 'housing', 'contact', 'day',
            'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y'
        ]
        self.X = None
        self.y = None

    def select_features(self):
        """Selects the specified features from the dataset."""
        self.df = self.df[self.features]

    def check_missing_values(self):
        """
        Checks for missing values in the dataset.

        Returns:
            pd.Series: A series containing the count of missing values for each feature.
        """
        missing_values = self.df.isnull().sum()
        return missing_values

    def encode_target(self):
        """Encodes the target variable 'y' by mapping 'yes' to 1 and 'no' to 0."""
        self.df['y'] = self.df['y'].map({'yes': 1, 'no': 0})

    def split_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """
        Splits the dataset into training, validation, and test sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the dataset to include in the validation split.
            random_state (int): Seed used by the random number generator.

        Returns:
            tuple: Training, validation, and test sets.
        """
        X = self.df.drop('y', axis=1)
        y = self.df['y']

        X_train_full, X_temp, y_train_full, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=random_state
        )

        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state
        )

        self.X = {'train': X_train_full, 'val': X_val, 'test': X_test}
        self.y = {'train': y_train_full, 'val': y_val, 'test': y_test}

        return self.X, self.y
