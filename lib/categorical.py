import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from lib.embedding_sum import Digitizer, EmbeddingSumClassifier


class ColumnEncoder:
    """
    For X of type pd.DataFrame only. Add custom order to the values. For object columns a default is used if not provided.
    """
    def __init__(self, category_order: dict):
        self.category_order = category_order
        self.category_columns = []
        
    def fit(self, X):
        # add default order for categorical columns outside self.category_order
        for obj_col in X.select_dtypes(["object"]):
            if obj_col not in self.category_order:
                self.category_order[obj_col] = X[obj_col].unique()
                
        # save indices of categorical columns and their sizes for later
        self.category_columns = [ix for ix, col in enumerate(X) if col in self.category_order]
        self.category_size = [len(self.category_order.get(col, [])) for col in X]
        return self

    def transform(self, X):
        X = X.copy()
        for obj_col, category in self.category_order.items():
            X[obj_col] = pd.Categorical(X[obj_col], categories=category, ordered=True).codes
        return np.array(X.astype("float"))


class CategoricalDigitizer(Digitizer):
    
    def __init__(self, max_bins: int, category_order: dict):
        self.column_encoder = ColumnEncoder(category_order)
        super().__init__(max_bins)

    @property
    def category_columns(self):
        return self.column_encoder.category_columns

    @property
    def category_size(self):
        return self.column_encoder.category_size
            
    def fit(self, X):
        self.column_encoder.fit(X)
        X = self.column_encoder.transform(X)
        
        cutoffs_with_weights = [
            self._get_cutoffs_and_weights_for_categorical(col_ix, X[:, col_ix])
            if col_ix in self.category_columns
            else self._get_cutoffs_and_weights_for_feature(X[:, col_ix])
            for col_ix in range(X.shape[1])
        ]
        self.cutoffs_, self.weights_ = map(list, zip(*(cutoffs_with_weights)))
        return self
    
    def transform(self, X):
        X = self.column_encoder.transform(X)
        X = np.array(X)
        digitized = np.empty_like(X)
        for col_ix, cutoffs in enumerate(self.cutoffs_):
            if col_ix in self.category_columns:
                digitized[:, col_ix] = X[:, col_ix]
            else:
                digitized[:, col_ix] = np.digitize(X[:, col_ix], cutoffs, right=True)
        return digitized
        
    def _get_cutoffs_and_weights_for_categorical(self, col_ix, x):
        category_size = self.category_size[col_ix]
        cutoffs = np.arange(0, category_size-1)
        unique, counts = np.unique(x, return_counts=True)
        existing_counts = dict(zip(unique, counts))
        weights = np.array([existing_counts.get(ix, 0) / len(x) for ix in range(category_size)])
        return cutoffs, weights


class CategoricalEmbeddingSumClassifier(EmbeddingSumClassifier):
    
    def __init__(
            self,
            max_bins: int,
            category_order: dict,
            max_epochs: int,
            lr: float,
            step_loss_weight: float,
            embedding_sum_loss_weight: float,
    ):
        super().__init__(max_bins=max_bins, max_epochs=max_epochs, lr=lr, step_loss_weight=step_loss_weight, embedding_sum_loss_weight=embedding_sum_loss_weight)
        self.digitizer = CategoricalDigitizer(max_bins=max_bins, category_order=category_order)

    def feature_importance(self, feature_names: list[str]) -> dict:
        result = []
        assert len(feature_names) == len(self.module_.embeddings)
        for i, feature_name in enumerate(feature_names):
            embedding_values = self.module_.embeddings[i].weight.data.detach().numpy()[:, 0]
            is_categorical = i in self.digitizer.category_columns
            if is_categorical:
                values = self.digitizer.column_encoder.category_order[feature_name]
            else:
                values = np.concatenate([self.digitizer.cutoffs_[i], [np.inf]])
            for val, weight in zip(values, embedding_values):
                result.append([feature_name, val, weight, is_categorical])

        return pd.DataFrame(result, columns=["feature", "value", "impact", "categorical"])

    def visualize(self, feature_names: list[str], subset: list[str] = None):
        """
        Overwrite visualisation of category features.
        """
        assert len(feature_names) == len(self.module_.embeddings)
        max_embedding_abs = max(
            max(np.abs(emb.weight.data.detach().numpy()[:, 0]))
            for emb in self.module_.embeddings
        )
        for i, feature_name in enumerate(feature_names):
            if subset is not None and feature_name not in subset:
                continue

            if i in self.digitizer.category_columns:
                # data
                embedding_values = self.module_.embeddings[i].weight.data.detach().numpy()[:, 0]
                widths = self.digitizer.weights_[i]
                xs = np.concatenate([[0], widths.cumsum()])
    
                fig, ax = plt.subplots()
                ax.set_title(feature_name)
                ax.set_xlim((0, 1))
                bars = ax.bar(
                    x=xs[:-1],
                    width=widths,
                    height=embedding_values,
                    align='edge',
                    color=np.where(embedding_values > 0, 'red', 'green'),
                    edgecolor='black',
                )
                categories = self.digitizer.column_encoder.category_order[feature_name]
                ax.bar_label(bars, labels=categories)
                ax.set_ylim((-max_embedding_abs - .1, max_embedding_abs + .1))
                ax.set_xticks([])
                ax.set_ylabel('embedding value')
            else:
                super().visualize(feature_names=feature_names, subset=[feature_name])

    def visualize_most_important(self, feature_names: list[str], top: int=5):
        features_df = self.feature_importance(feature_names=feature_names)
        features_to_visalize = (
            features_df.groupby(by="feature").agg({"impact": "max"}).head(top).index.tolist()
        )
        # visualize in the order of importance
        for feature_name in features_to_visalize:
            self.visualize(feature_names=feature_names, subset=[feature_name])
