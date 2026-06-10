import os
import numpy as np
import pytest

from numcompute_stream.io import read_csv


def test_read_csv_valid_file():
    # pytest is being run from project root
    file_path = os.path.join(os.getcwd(), "data", "Iris.csv")

    assert os.path.exists(file_path), "Dataset file does not exist"

    data = read_csv(file_path)

    assert isinstance(data, np.ndarray)
    assert data.ndim == 2
    assert data.shape[0] > 0
    assert data.shape[1] > 0


def test_read_csv_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_csv("non_existent_file.csv")


def test_read_csv_contains_missing_placeholder(tmp_path):
    file = tmp_path / "test.csv"
    file.write_text("1,2,\n3,4,5")

    data = read_csv(file)

    assert data[0, 2] == ""


def test_read_csv_string_handling(tmp_path):
    file = tmp_path / "test_strings.csv"
    file.write_text("A,B,C\nx,y,z")

    data = read_csv(file)

    assert isinstance(data[1, 0], str)


def test_read_csv_empty_file(tmp_path):
    file = tmp_path / "empty.csv"
    file.write_text("")

    data = read_csv(file)

    assert isinstance(data, np.ndarray)
    assert data.size == 0


def test_read_csv_mixed_types(tmp_path):
    file = tmp_path / "mixed.csv"
    file.write_text("1,hello,3\n4,world,6")

    data = read_csv(file)

    assert data.shape == (2, 3)