import numpy as np


class Pipeline:
    """
    Chain preprocessing transformers and a final estimator.

    Transformers must implement:
        fit(X)
        transform(X)

    Streaming transformers should implement:
        partial_fit(X)

    Final estimator must implement:
        fit(X, y)
        predict(X)

    Streaming estimators should implement:
        partial_fit(X, y)

    Parameters
    ----------
    steps : list of tuple
        List of (name, step) pairs. All steps except the final step are treated
        as transformers. The final step is treated as an estimator unless it also
        behaves like a transformer-only step.
    allow_fit_fallback : bool, default=True
        If True, fit() may be used when partial_fit() is missing. This keeps
        the pipeline compatible with simple non-streaming transformers while
        still allowing strict streaming checks when set to False.
    rebuild_on_feature_change : bool, default=True
        If True, the final estimator is rebuilt from accumulated raw chunks when
        transformer output dimensionality changes. This protects streaming
        pipelines that use incrementally expanding encoders.

    Raises
    ------
    ValueError
        If steps is empty, or required methods are missing.

    Complexity
    ----------
    fit: O(sum of step fit/transform costs)
    partial_fit: O(sum of step partial_fit/transform costs)
    predict: O(sum of transformer transform costs + estimator predict cost)
    """

    def __init__(self, steps, allow_fit_fallback=True, rebuild_on_feature_change=True):
        if not steps:
            raise ValueError("Pipeline requires at least one step")

        self.steps = steps
        self.allow_fit_fallback = allow_fit_fallback
        self.rebuild_on_feature_change = rebuild_on_feature_change

        self._X_memory = None
        self._y_memory = None
        self._last_n_features = None

    def _call_with_optional_kwargs(self, func, *args, **kwargs):
        """
        Call a method with keyword arguments, falling back if unsupported.
        """
        try:
            return func(*args, **kwargs)
        except TypeError:
            return func(*args)

    def _fit_transformers(self, X):
        """
        Fit and transform all transformer steps except the final estimator.
        """
        X_current = X

        for name, step in self.steps[:-1]:
            if not hasattr(step, "fit") or not hasattr(step, "transform"):
                raise ValueError(f"Step '{name}' must implement fit and transform")

            step.fit(X_current)
            X_current = step.transform(X_current)

        return X_current

    def _partial_fit_transformers(self, X):
        """
        Incrementally update and transform all transformer steps.
        """
        X_current = X

        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise ValueError(f"Step '{name}' must implement transform")

            if hasattr(step, "partial_fit"):
                step.partial_fit(X_current)
            elif self.allow_fit_fallback and hasattr(step, "fit"):
                step.fit(X_current)
            else:
                raise ValueError(f"Step '{name}' must implement partial_fit for streaming use")

            X_current = step.transform(X_current)

        return X_current

    def _transformers_only(self, X):
        """
        Apply only the transformer steps before the final estimator.
        """
        X_current = X

        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise ValueError(f"Step '{name}' must implement transform")

            X_current = step.transform(X_current)

        return X_current

    def _store_raw_chunk(self, X, y):
        """
        Store raw chunks so the final estimator can be rebuilt if dimensions expand.
        """
        X_arr = np.asarray(X, dtype=object)
        y_arr = None if y is None else np.asarray(y)

        if y_arr is None:
            return

        if self._X_memory is None:
            self._X_memory = X_arr.copy()
            self._y_memory = y_arr.copy()
        else:
            self._X_memory = np.vstack([self._X_memory, X_arr])
            self._y_memory = np.concatenate([self._y_memory, y_arr])

    def _fit_final_estimator(self, X, y=None, **fit_params):
        """
        Fit the final estimator, passing optional parameters when supported.
        """
        last_name, last_step = self.steps[-1]

        if not hasattr(last_step, "fit"):
            raise ValueError(f"Final step '{last_name}' must implement fit")

        if y is not None:
            self._call_with_optional_kwargs(last_step.fit, X, y, **fit_params)
        else:
            self._call_with_optional_kwargs(last_step.fit, X, **fit_params)

    def _partial_fit_final_estimator(self, X, y=None, **fit_params):
        """
        Incrementally update the final estimator.
        """
        last_name, last_step = self.steps[-1]

        if hasattr(last_step, "partial_fit"):
            if y is not None:
                self._call_with_optional_kwargs(last_step.partial_fit, X, y, **fit_params)
            else:
                self._call_with_optional_kwargs(last_step.partial_fit, X, **fit_params)
        elif self.allow_fit_fallback and hasattr(last_step, "fit"):
            if y is not None:
                self._call_with_optional_kwargs(last_step.fit, X, y, **fit_params)
            else:
                self._call_with_optional_kwargs(last_step.fit, X, **fit_params)
        else:
            raise ValueError(f"Final step '{last_name}' must implement partial_fit for streaming use")

    def fit(self, X, y=None, **fit_params):
        """
        Fit the pipeline on a full batch of data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        y : array-like of shape (n_samples,), optional
            Target labels for supervised estimators.
        **fit_params
            Extra keyword arguments passed to the final estimator. For example,
            classes=... for classifiers whose first chunk may miss classes.

        Returns
        -------
        self : Pipeline
        """
        self._X_memory = None
        self._y_memory = None
        self._last_n_features = None

        X_current = self._fit_transformers(X)
        self._fit_final_estimator(X_current, y, **fit_params)
        self._last_n_features = X_current.shape[1] if hasattr(X_current, "shape") else None
        self._store_raw_chunk(X, y)

        return self

    def partial_fit(self, X, y=None, **fit_params):
        """
        Incrementally update the pipeline using one data chunk.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature chunk.
        y : array-like of shape (n_samples,), optional
            Target chunk.
        **fit_params
            Extra keyword arguments passed to the final estimator. This supports
            calls such as pipe.partial_fit(X_chunk, y_chunk, classes=classes).

        Returns
        -------
        self : Pipeline

        Raises
        ------
        ValueError
            If transformer steps do not support partial_fit, if transform is
            missing, or if the final estimator cannot be updated.

        Notes
        -----
        Transformer steps are updated first, then the transformed chunk is passed
        to the final estimator. If transformer output dimensionality expands
        because of a streaming encoder, the final estimator is rebuilt from the
        accumulated raw chunks when rebuild_on_feature_change=True.
        """
        X_current = self._partial_fit_transformers(X)
        self._store_raw_chunk(X, y)

        current_n_features = X_current.shape[1] if hasattr(X_current, "shape") else None
        feature_changed = (
            self._last_n_features is not None
            and current_n_features is not None
            and current_n_features != self._last_n_features
        )

        if feature_changed and self.rebuild_on_feature_change and y is not None:
            X_all = self._transformers_only(self._X_memory)
            last_step = self.steps[-1][1]

            if hasattr(last_step, "reset"):
                last_step.reset()

            self._fit_final_estimator(X_all, self._y_memory, **fit_params)
            self._last_n_features = X_all.shape[1]
            return self

        self._partial_fit_final_estimator(X_current, y, **fit_params)
        self._last_n_features = current_n_features

        return self

    def transform(self, X):
        """
        Apply transformer steps to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_current : array-like
            Transformed data.
        """
        # If the pipeline contains only a transformer, transform through it.
        if len(self.steps) == 1 and hasattr(self.steps[0][1], "transform"):
            return self.steps[0][1].transform(X)

        return self._transformers_only(X)

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit the pipeline and return transformed data.
        """
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def predict(self, X):
        """
        Predict using the final estimator after applying transformer steps.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
        """
        X_current = self._transformers_only(X)
        last_name, last_step = self.steps[-1]

        if not hasattr(last_step, "predict"):
            raise ValueError(f"Final step '{last_name}' must implement predict")

        return last_step.predict(X_current)

    def predict_proba(self, X):
        """
        Predict class probabilities using the final estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
        """
        X_current = self._transformers_only(X)
        last_name, last_step = self.steps[-1]

        if not hasattr(last_step, "predict_proba"):
            raise ValueError(f"Final step '{last_name}' must implement predict_proba")

        return last_step.predict_proba(X_current)

    def score(self, X, y):
        """
        Compute accuracy score using the final estimator.
        """
        y_pred = self.predict(X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(y) != len(y_pred):
            raise ValueError("X and y must contain the same number of samples")

        return np.mean(y == y_pred)
