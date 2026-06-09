import numpy as np
import warnings


# reads a CSV file and converts it into a numpy array
def read_csv(file_path, delimiter=",", dtype=object, missing_val=""):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        # using genfromtxt so it can handle missing values

        arr = np.genfromtxt(
            file_path,
            delimiter=delimiter,
            dtype=dtype,
            missing_values=missing_val,
            filling_values=np.nan,
            encoding="utf-8"
        )

        # handle empty files
        if arr.size == 0:
            return arr

        return np.vectorize(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x,
            otypes=[object]
        )(arr)