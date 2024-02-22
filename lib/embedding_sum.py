import collections
import dataclasses
from typing import Any, Hashable, Protocol, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class Digitization(Protocol):

    def digitize(self, x: np.array) -> np.array:
        raise NotImplementedError()

    def is_categorical(self) -> bool:
        raise NotImplementedError()

    def is_continuous(self) -> bool:
        raise NotImplementedError()

    def get_cutoffs(self) -> np.ndarray | None:
        raise NotImplementedError()

    def get_categorical_values(self) -> list | None:
        raise NotImplementedError()

    def get_weights(self) -> np.ndarray:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class ContinuousDigitization(Digitization):
    cutoffs: np.ndarray
    weights: np.ndarray

    def digitize(self, x: np.array) -> np.ndarray:
        return np.digitize(x, self.cutoffs, right=True)

    def is_categorical(self) -> bool:
        return False

    def is_continuous(self) -> bool:
        return True

    def get_cutoffs(self) -> np.ndarray | None:
        return self.cutoffs

    def get_categorical_values(self) -> list | None:
        return None

    def get_weights(self) -> np.array:
        return self.weights

    @staticmethod
    def from_values(x: np.array, *, min_weight: float, max_bins: float):
        initial_cutoffs = np.unique([
            np.quantile(x, q, method='lower')
            for q in np.arange(1, max_bins) / max_bins
        ])
        final_cutoffs, weights = ContinuousDigitization._merge_narrow_bins(x, initial_cutoffs, min_weight=min_weight)
        return ContinuousDigitization(cutoffs=final_cutoffs, weights=weights)

    @staticmethod
    def _merge_narrow_bins(x, cutoffs, *, min_weight: float):
        weights = ContinuousDigitization._get_weights_from_cutoffs(x, cutoffs)
        # this implementation favors legibility over efficiency
        while np.min(weights) < min_weight:
            ix = np.argmin(weights)
            neighbor_weight_and_border_cutoff_ix_pairs = []
            if 0 < ix:
                neighbor_weight_and_border_cutoff_ix_pairs.append((weights[ix - 1], ix - 1))
            if ix + 1 < len(weights):
                neighbor_weight_and_border_cutoff_ix_pairs.append((weights[ix + 1], ix))
            assert neighbor_weight_and_border_cutoff_ix_pairs
            smallest_neighbor_weight, cutoff_to_delete = min(neighbor_weight_and_border_cutoff_ix_pairs)
            cutoffs = np.delete(cutoffs, cutoff_to_delete)
            weights = ContinuousDigitization._get_weights_from_cutoffs(x, cutoffs)
        return cutoffs, weights

    @staticmethod
    def _get_weights_from_cutoffs(x, cutoffs):
        return np.diff([0, *(np.mean(x <= cutoff) for cutoff in cutoffs), 1])


@dataclasses.dataclass(frozen=True)
class CategoricalDigitization(Digitization):
    values: list
    weights: np.ndarray

    def digitize(self, x: np.array) -> np.array:
        return np.array([self.values.index(c) for c in x], dtype=np.int32)

    def is_categorical(self) -> bool:
        return True

    def is_continuous(self) -> bool:
        return False

    def get_cutoffs(self) -> np.ndarray | None:
        return None

    def get_categorical_values(self) -> list | None:
        return self.values

    def get_weights(self) -> np.ndarray:
        return self.weights

    @staticmethod
    def from_values(x: np.array, *, values: list):
        assert len(set(values)) == len(values), 'values should be unique'
        counter = collections.Counter(x)
        return CategoricalDigitization(
            values=values,
            weights=np.array([counter[v] for v in values]),
        )


class Digitizer:

    def __init__(self, max_bins: int):
        self.max_bins = max_bins
        self.min_weight = (1 / max_bins) / 2

        self.digitizations_: list[Digitization] | None = None

    def fit(self, X, *, categorical_values: dict[int, list] | None):
        X = np.array(X)
        digitizations = []

        for col_ix in range(X.shape[1]):
            x = X[:, col_ix]
            if categorical_values is not None and col_ix in categorical_values:
                digitization = CategoricalDigitization.from_values(
                    x,
                    values=categorical_values[col_ix],
                )
            else:
                digitization = ContinuousDigitization.from_values(
                    x,
                    max_bins=self.max_bins,
                    min_weight=self.min_weight,
                )
            digitizations.append(digitization)

        self.digitizations_ = digitizations
        return self

    def transform(self, X):
        X = np.array(X)
        digitized = np.empty_like(X, dtype=np.int32)
        for col_ix, digitization in enumerate(self.digitizations_):
            digitized[:, col_ix] = digitization.digitize(X[:, col_ix])
        return digitized


class EmbeddingSumModule(nn.Module):

    def __init__(
            self,
            values_weights: Sequence[Sequence[float]],
            free_term,
            *,
            is_categorical: list[bool],
            dtype=torch.float32,
    ):
        super().__init__()
        self.values_weights = [
            torch.tensor(w, dtype=dtype, requires_grad=False)
            for w in values_weights
        ]
        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=len(w),
                embedding_dim=1,
                _weight=torch.zeros((len(w), 1), dtype=dtype, requires_grad=True),
            )
            for w in values_weights
        ])
        self.free_term = nn.Parameter(
            data=free_term.clone().detach(),
            requires_grad=True,
        )
        assert len(is_categorical) == len(values_weights)
        self.is_categorical = is_categorical

        self.dtype = dtype

    def forward(self, X):
        result = self.free_term.expand(len(X))
        assert X.shape[1] == len(self.embeddings)
        for i, emb in enumerate(self.embeddings):
            result = result + emb(X[:, i]).flatten()
        return result

    def mean_square_step(self):
        if all(self.is_categorical):
            return torch.tensor(0)
        return torch.concat([
            (emb.weight[1:] - emb.weight[:-1]).flatten()
            for emb, is_categorical in zip(self.embeddings, self.is_categorical)
            if not is_categorical
        ]).pow(2).mean()

    def mean_square_embedding_sum(self):
        return torch.concat([
            ((w @ emb.weight).sum() / w.sum()).view(1)
            for w, emb in zip(self.values_weights, self.embeddings)
        ]).pow(2).mean()

    def mean_square_category_embedding(self):
        if not any(self.is_categorical):
            return torch.tensor(0)
        return torch.concat([
            emb.weight.flatten()
            for emb, is_categorical in zip(self.embeddings, self.is_categorical)
            if is_categorical
        ]).pow(2).mean()


@dataclasses.dataclass(frozen=True)
class CategoricalFeatureInfo:
    values: list[Hashable]
    value_names: dict[Hashable, str]


@dataclasses.dataclass(frozen=True)
class FeaturesInfo:
    feature_names: list[str] | None = None
    categorical_info: dict[int, CategoricalFeatureInfo] = None

    def get_categorical_feature_info(self, key: str | int) -> CategoricalFeatureInfo | None:
        if self.categorical_info is None:
            return None
        if isinstance(key, str):
            assert self.feature_names is not None
            key = self.feature_names.index(key)
        return self.categorical_info.get(key)

    @staticmethod
    def from_categorical_value_names(
            feature_names: list[str] | None = None,
            categorical_value_names: dict[str, dict[Hashable, str]] | None = None,
    ):
        if categorical_value_names is None:
            categorical_value_names = {}
        categorical_info = {}
        if feature_names is None:
            assert len(categorical_value_names) == 0
            return FeaturesInfo()
        for feature_name, value_names in categorical_value_names.items():
            categorical_info[feature_names.index(feature_name)] = CategoricalFeatureInfo(
                values=list(value_names),
                value_names=value_names,
            )
        return FeaturesInfo(feature_names, categorical_info)


class EmbeddingSumClassifier:
    """ Scikit-learn-compatible classifier """

    def __init__(
            self,
            *,
            max_bins: int,
            max_epochs: int,
            lr: float,
            step_loss_weight: float,
            embedding_sum_loss_weight: float,
            category_embedding_loss_weight: float,
    ):
        super().__init__()
        self.digitizer = Digitizer(max_bins=max_bins)
        self.max_epochs = max_epochs
        self.lr = lr
        self.step_loss_weight = step_loss_weight
        self.embedding_sum_loss_weight = embedding_sum_loss_weight
        self.category_embedding_loss_weight = category_embedding_loss_weight

        self.classes_ = np.array([0, 1])  # sklearn compatibility
        self.features_info_: FeaturesInfo | None = None
        self.module_: EmbeddingSumModule | None = None
        self.training_history_: list[dict[str, Any]] | None = None

    def fit(  # sklearn compatibility
            self,
            X, y,
            weight=None,

            # features info or its components
            features_info: FeaturesInfo | None = None,
            feature_names: list[str] | None = None,
            categorical_value_names: dict[str, dict[Hashable, str]] | None = None,
    ):
        if features_info is None:
            if feature_names is None and isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
            features_info = FeaturesInfo.from_categorical_value_names(
                feature_names=feature_names,
                categorical_value_names=categorical_value_names,
            )
        self.features_info_ = features_info
        self.digitizer.fit(
            X,
            categorical_values={
                i: features_info.get_categorical_feature_info(i).values
                for i in range(X.shape[1])
                if features_info.get_categorical_feature_info(i) is not None
            },
        )
        X_tensor = torch.tensor(self.digitizer.transform(X), dtype=torch.int32)
        y_tensor = torch.tensor(np.array(y), dtype=torch.float32)
        weight_tensor = None if weight is None else torch.tensor(np.array(weight), dtype=torch.float32)
        self.module_ = EmbeddingSumModule(
            values_weights=[d.get_weights() for d in self.digitizer.digitizations_],
            free_term=torch.logit(torch.mean(y_tensor)),
            is_categorical=[d.is_categorical() for d in self.digitizer.digitizations_],
        )
        self.module_.train()
        with torch.enable_grad():
            self.train(X_tensor, y_tensor, weight=weight_tensor)
        self.module_.eval()
        return self

    def predict_proba(self, X):  # sklearn compatibility
        X_tensor = torch.tensor(self.digitizer.transform(X), dtype=torch.int32)
        with torch.no_grad():
            y_pred = F.sigmoid(self.module_(X_tensor)).numpy()
        result = np.empty(shape=(X.shape[0], 2))
        result[:, 0] = 1 - y_pred
        result[:, 1] = y_pred
        return result

    def train(self, X, y_true, weight=None):
        m = self.module_
        optimizer = torch.optim.SGD(params=m.parameters(), lr=self.lr)
        self.training_history_ = []
        for epoch in range(self.max_epochs):
            m.zero_grad()
            y_pred = m(X)
            train_clf_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=weight)
            train_step_loss = m.mean_square_step()
            train_embedding_sum_loss = m.mean_square_embedding_sum()
            train_category_embedding_loss = m.mean_square_category_embedding()
            train_loss = (
                    train_clf_loss
                    + train_step_loss * self.step_loss_weight
                    + train_embedding_sum_loss * self.embedding_sum_loss_weight
                    + train_category_embedding_loss * self.category_embedding_loss_weight
            )
            train_loss.backward()
            optimizer.step()
            self.training_history_.append({
                'epoch': epoch,
                'train_loss': train_loss.detach().item(),
                'train_clf_loss': train_clf_loss.detach().item(),
                'train_step_loss': train_step_loss.detach().item(),
                'train_embedding_sum_loss': train_embedding_sum_loss.detach().item(),
                'train_category_embedding_loss': train_category_embedding_loss.detach().item(),
            })

    def plot_training_history(self):
        (
            pd.DataFrame(self.training_history_)
            .assign(
                train_step_loss_weighted=lambda df: df['train_step_loss'] * self.step_loss_weight,
                train_embedding_sum_loss_weighted=lambda df: (
                    df['train_embedding_sum_loss'] * self.embedding_sum_loss_weight
                ),
                train_category_embedding_loss_weighted=lambda df: (
                    df['train_category_embedding_loss'] * self.category_embedding_loss_weight
                ),
            )
            .set_index('epoch')
            .plot()
        )

    def visualize(self, subset: list[str] = None):
        feature_names = (
            self.features_info_.feature_names
            or list(map('feature {}'.format, range(len(self.module_.embeddings))))
        )
        assert len(feature_names) == len(self.module_.embeddings)
        max_embedding_abs = max(
            max(np.abs(emb.weight.data.detach().numpy()[:, 0]))
            for emb in self.module_.embeddings
        )
        for i, feature_name in enumerate(feature_names):
            if subset is not None and feature_name not in subset:
                continue

            # data
            digitization = self.digitizer.digitizations_[i]
            embedding_values = self.module_.embeddings[i].weight.data.detach().numpy()[:, 0]
            widths = digitization.get_weights()
            xs = np.concatenate([[0], widths.cumsum()])

            fig, ax = plt.subplots()
            ax.set_title(feature_name)
            for x in xs[1:-1]:
                ax.axvline(x, color='grey', linestyle=':', linewidth=1)
            ax.set_xlim((xs[0], xs[-1]))
            bars = ax.bar(
                x=xs[:-1],
                width=widths,
                height=embedding_values,
                align='edge',
                color=np.where(embedding_values > 0, 'red', 'green')
            )
            ax.axhline(0, color='grey', linewidth=1)
            ax.set_ylim((-max_embedding_abs - .1, max_embedding_abs + .1))
            ax.set_xticks([])
            ax.set_ylabel('embedding value')

            if digitization.is_continuous():
                ax2 = ax.twinx()
                ax2.scatter(x=xs[1:-1], y=digitization.get_cutoffs(), color='royalblue')
                ax2.tick_params(axis='y', labelcolor='royalblue')
                ax2.set_ylabel('cutoff', color='royalblue')
            elif digitization.is_categorical():
                categorical_feature_info: CategoricalFeatureInfo = self.features_info_.get_categorical_feature_info(i)
                ax.bar_label(bars, labels=[
                    categorical_feature_info.value_names[v]
                    for v in digitization.get_categorical_values()
                ])

