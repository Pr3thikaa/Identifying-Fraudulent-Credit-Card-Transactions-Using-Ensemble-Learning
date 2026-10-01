import pandas as pd
from src.preprocess import clean_data


def test_removes_duplicates():
    df = pd.DataFrame({"Amount": [10, 10, 5], "Class": [0, 0, 1]})
    assert len(clean_data(df)) == 2


def test_no_nulls():
    df = pd.DataFrame({"Amount": [10, None], "Class": [0, 1]})
    assert clean_data(df).isnull().sum().sum() == 0


def test_labels_binary():
    df = pd.DataFrame({"Amount": [1, 2], "Class": [0, 1]})
    assert set(clean_data(df)["Class"]) <= {0, 1}