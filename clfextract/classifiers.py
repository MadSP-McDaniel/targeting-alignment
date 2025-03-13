from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.contingency_tables import mcnemar
from torch.utils.data import DataLoader, TensorDataset


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class Classifier:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        criterion: Optional[nn.Module] = None,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.model = model.to(self.device)
        self.criterion = criterion or nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.verbose = verbose

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        num_epochs: int = 100,
        n_splits: int = 1,
        patience: Optional[int] = None,
    ) -> List[float]:
        if n_splits > 1:
            return self._train_with_cv(X, y, num_epochs, n_splits, patience)
        else:
            return self._train_simple(X, y, num_epochs, patience)

    def _train_simple(
        self, X: torch.Tensor, y: torch.Tensor, num_epochs: int, patience: Optional[int]
    ) -> List[float]:
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        early_stopping = EarlyStopping(patience=patience) if patience else None

        loss_history = []
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = self._train_epoch(dataloader)

            loss_history.append(epoch_loss)
            if self.verbose:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

            if early_stopping:
                early_stopping(epoch_loss)
                if early_stopping.early_stop:
                    if self.verbose:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        return loss_history

    def _train_with_cv(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        num_epochs: int,
        n_splits: int,
        patience: Optional[int],
        random_state: Optional[int] = None,
    ) -> List[float]:
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        loss_history = []

        for fold, (train_index, val_index) in enumerate(
            skf.split(X.detach().cpu(), y.detach().cpu())
        ):
            if self.verbose:
                print(f"Fold {fold + 1}/{n_splits}")
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

            early_stopping = EarlyStopping(patience=patience) if patience else None

            for epoch in range(num_epochs):
                self.model.train()
                train_loss = self._train_epoch(train_loader)

                self.model.eval()
                val_loss = self._validate(val_loader)

                loss_history.append((train_loss, val_loss))
                if self.verbose:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                    )

                if early_stopping:
                    early_stopping(val_loss)
                    if early_stopping.early_stop:
                        if self.verbose:
                            print(f"Early stopping triggered at epoch {epoch + 1}")
                        break

            if fold < n_splits - 1:
                # Reset model for next fold
                self.model.apply(self._weight_reset)

        return loss_history

    def _train_epoch(self, dataloader):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs.squeeze(), batch_y.float())
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def _validate(self, dataloader):
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y.float())
                val_loss += loss.item()

        return val_loss / len(dataloader)

    def _weight_reset(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    def predict(self, X: torch.Tensor, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Make predictions using the trained classifier.

        Args:
            X (torch.Tensor): The input features.

        Returns:
            np.ndarray: The predicted probabilities.
        """
        if batch_size is None:
            batch_size = self.batch_size
        dataloader = DataLoader(X, batch_size=self.batch_size)
        predictions = []

        with torch.no_grad():
            for batch_X in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.extend(outputs.cpu().numpy())

        return np.array(predictions)

    def evaluate(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        threshold: float = 0.5,
        batch_size: Optional[int] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate the classifier

        Args:
            X (torch.Tensor): The input features.
            y (torch.Tensor): The true labels.

        Returns:
            Tuple[float, np.ndarray, np.ndarray]: The accuracy and F1 score
        """
        batch_size = batch_size if batch_size is not None else self.batch_size
        y_pred = self.predict(X, batch_size=batch_size).squeeze()
        y_true = y.cpu().numpy().squeeze()
        accuracy = ((y_pred > threshold) == y_true).mean()
        f1 = f1_score(y_true, y_pred > threshold)

        return accuracy, f1

    def roc_curve(self, X: torch.Tensor, y: torch.Tensor):
        """
        Compute the ROC curve.

        Args:
            X (torch.Tensor): The input features.
            y (torch.Tensor): The true labels.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The false positive rate, true positive rate, and thresholds.
        """
        y_pred = self.predict(X)
        fpr, tpr, thresholds = roc_curve(y.cpu().numpy(), y_pred)
        return fpr, tpr, thresholds

    def precision_recall_curve(self, X: torch.Tensor, y: torch.Tensor):
        """
        Compute the precision-recall curve.

        Args:
            X (torch.Tensor): The input features.
            y (torch.Tensor): The true labels.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The precision, recall, and thresholds.
        """
        y_pred = self.predict(X)
        precision, recall, thresholds = precision_recall_curve(y.cpu().numpy(), y_pred)
        return precision, recall, thresholds

    def get_all_metrics(self, X: torch.Tensor, y: torch.Tensor, threshold: float = 0.5):
        """
        Compute all evaluation metrics.

        Args:
            X (torch.Tensor): The input features.
            y (torch.Tensor): The true labels.

        Returns:
            Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                The accuracy, F1 score, false positive rate, true positive rate, precision, recall, and thresholds.
        """
        accuracy, f1 = self.evaluate(X, y, threshold)
        fpr, tpr, roc_thresholds = self.roc_curve(X, y)
        precision, recall, pr_thresholds = self.precision_recall_curve(X, y)

        results = {
            "accuracy": accuracy,
            "f1": f1,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "roc_thresholds": roc_thresholds.tolist(),
            "pr_thresholds": pr_thresholds.tolist(),
        }

        return results

    def save(self, path: str):
        """
        Save the model to a file.

        Args:
            path (str): The file path.
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """
        Load the model from a file.

        Args:
            path (str): The file path.
        """
        self.model.load_state_dict(torch.load(path))


class LinearClassifier(Classifier):
    def __init__(self, input_dim: int, **kwargs):
        model = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())
        super().__init__(model, **kwargs)


class MLPClassifier(Classifier):
    def __init__(
        self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, **kwargs
    ):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())
        model = nn.Sequential(*layers)
        super().__init__(model, **kwargs)


class RNNClassifier(Classifier):
    def __init__(
        self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, **kwargs
    ):
        model = RNNModel(input_dim, hidden_dim, num_layers)
        super().__init__(model, **kwargs)


class RNNModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.linear(h_n[-1])
        return self.sigmoid(out)


def compare_classifiers(
    setting1: Union[Tuple[Classifier, torch.Tensor], torch.Tensor],
    setting2: Union[Tuple[Classifier, torch.Tensor], torch.Tensor],
    method: str = "agreement",
    threshold: float = 0.5,
) -> Union[float, Tuple[float, float]]:
    """
    Compare two fitted classifiers using either agreement rate or McNemar's test.

    Args:
        setting1 (Union[Tuple[Classifier, torch.Tensor],torch.Tensor]): Can be y_pred or (clf, X)
        method (str): The comparison method, either 'agreement' or 'mcnemar'. Defaults to 'agreement'.

    Returns:
        Union[float, Tuple[float, float]]:
            If method is 'agreement': The agreement rate between the two classifiers.
            If method is 'mcnemar': The test statistic and p-value of McNemar's test.

    Raises:
        ValueError: If an invalid method is specified.
    """
    if isinstance(setting1, tuple):
        clf1, X1 = setting1
        y_pred1 = (clf1.predict(X1) > threshold).astype(int)
    else:
        y_pred1 = setting1.cpu().numpy().astype(int)

    if isinstance(setting2, tuple):
        clf2, X2 = setting2
        y_pred2 = (clf2.predict(X2) > threshold).astype(int)
    else:
        y_pred2 = setting2.cpu().numpy().astype(int)

    if method == "agreement":
        agreement_rate = (y_pred1 == y_pred2).mean()
        return agreement_rate
    elif method == "mcnemar":
        contingency_table = np.zeros((2, 2))
        for i in range(4):
            contingency_table[i // 2, i % 2] = np.sum(
                (y_pred1 == (i // 2)) & (y_pred2 == (i % 2))
            )
        test = mcnemar(contingency_table, exact=False, correction=True)
        return test.statistic, test.pvalue

    else:
        raise ValueError("Invalid method. Choose either 'agreement' or 'mcnemar'.")
