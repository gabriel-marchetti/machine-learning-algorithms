# The base of this code is inspired by the code developed by GiovaniValdrighi in his repository
# More can be found at: https://github.com/hiaac-finance/multiplicity/tree/master/multiplicity
from abc import ABC, abstractmethod
from typing import Optional
from sklearn.model_selection import train_test_split
import pandas as pd

class Dataset(ABC):
    def __init__(
        self, 
        name : str, 
        path : str,
        test_size : float = 0.2,
        random_state : Optional[int] = None
    ) -> None:
        self.name = name
        self.path = path
        self.random_state = random_state
        self.test_size = test_size

        self.data = None
        self.X_train = None
        self.y_train = None 
        self.X_test = None 
        self.y_test = None

        self.load_data()
        self.train_test_split(test_size=test_size, random_state=random_state)

    @abstractmethod
    def load_data(self) -> None:
        pass

    def train_test_split(self, test_size: float, random_state: Optional[int]) -> None:
        df: pd.DataFrame = self.data
        self.X = df.drop(columns=['Target'])
        self.y = df['Target']

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        ) = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

    def get_train_data(self):
        return self.X_train, self.y_train
    
    def get_test_data(self):
        return self.X_test, self.y_test

class German(Dataset):
    def __init__(
        self,
        path : str,
        random_state : Optional[int] = None
    ):
        super().__init__("German", path, random_state)
    
    def load_data(self) -> None:
        self.data = pd.read_csv(self.path)