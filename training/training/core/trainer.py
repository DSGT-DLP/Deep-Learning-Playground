from dataclasses import dataclass
import time
from typing import Iterator, TypeVar
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from training.core.criterion import CriterionHandler


@dataclass
class EpochResult:
    epoch_num: int
    train_loss: float
    test_loss: float
    epoch_time: float

    def __str__(self) -> str:
        return f"epoch: {self.epoch_num}, train loss: {self.train_loss}, test loss: {self.test_loss}"


T = TypeVar("T", bound=EpochResult)


class Trainer(Iterator[T]):
    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterionHandler: CriterionHandler,
        epochs: int,
    ):
        self.epoch_results: list[EpochResult] = []  # store results for each epoch
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.criterionHandler = criterionHandler
        self.epochs = epochs
        self.curr_epoch = 0

        if train_loader.batch_size == None:
            raise Exception("train_loader batch_size cannot be None")
        if test_loader.batch_size == None:
            raise Exception("test_loader batch_size cannot be None")
        self.num_train_batches = len(train_loader)
        # total number of data points used for training per epoch
        self.epoch_train_size = train_loader.batch_size * self.num_train_batches
        self.num_test_batches = len(test_loader)
        # total number of data points used for testing per epoch
        self.epoch_test_size = test_loader.batch_size * self.num_test_batches

    def _train_init(self):
        self._epoch_batch_loss = 0  # cumulative training/testing loss per epoch
        self._start_time = time.time()
        self.model.train(True)

    def _train_step(self, inputs: torch.Tensor, labels: torch.Tensor):
        self.optimizer.zero_grad()  # zero out gradient for each batch
        self.model.forward(inputs)  # make prediction on input
        self._outputs: torch.Tensor = self.model(inputs)  # make prediction on input
        loss = self.criterionHandler.compute_loss(self._outputs, labels)
        loss.backward()  # backpropagation
        self.optimizer.step()  # adjust optimizer weights
        self._epoch_batch_loss += float(loss.detach())

    def _train_end(self):
        self._epoch_time = time.time() - self._start_time
        self._mean_train_loss = self._epoch_batch_loss / self.num_train_batches

    def _test_init(self):
        self.model.train(False)  # test the model on test set
        self._epoch_batch_loss = 0

    def _test_step(self, inputs: torch.Tensor, labels: torch.Tensor):
        self._test_pred: torch.Tensor = self.model(inputs)
        loss = self.criterionHandler.compute_loss(self._test_pred, labels)
        self._epoch_batch_loss += float(loss.detach())

    def _test_end(self):
        self._mean_test_loss = self._epoch_batch_loss / self.num_test_batches

    def _get_epoch_result(self):
        return EpochResult(
            self.curr_epoch,
            self._mean_train_loss,
            self._mean_test_loss,
            self._epoch_time,
        )

    def __next__(self):
        if self.curr_epoch < self.epochs:
            self._train_init()
            for _, (inputs, labels) in enumerate(self.train_loader):
                self._train_step(inputs, labels)
            self._train_end()
            self._test_init()
            for _, (inputs, labels) in enumerate(self.test_loader):
                self._test_step(inputs, labels)
            self._test_end()
            epoch_result = self._get_epoch_result()
            self.epoch_results.append(epoch_result)
            self.curr_epoch += 1
            return epoch_result
        else:
            raise StopIteration


@dataclass
class ClassificationEpochResult(EpochResult):
    epoch_num: int
    train_loss: float
    test_loss: float
    epoch_time: float
    train_accuracy: float
    test_accuracy: float

    def __init__(
        self,
        epoch_num: int,
        train_loss: float,
        test_loss: float,
        epoch_time: float,
        train_accuracy: float,
        test_accuracy: float,
    ):
        super().__init__(epoch_num, train_loss, test_loss, epoch_time)
        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy

    def __str__(self) -> str:
        return f"{super().__str__()} train_acc: {self.train_accuracy}, val_acc: {self.test_loss}"


class ClassificationTrainer(Trainer[ClassificationEpochResult]):
    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterionHandler: CriterionHandler,
        epochs: int,
        category_list: list[str],
    ):
        super().__init__(
            train_loader,
            test_loader,
            model,
            optimizer,
            criterionHandler,
            epochs,
        )
        self.labels_last_epoch = []
        self.y_pred_last_epoch = []
        self.category_list = category_list

    def compute_correct(self, predicted, actual):
        """
        Given a prediction (usually in logit form for classification problem), count the number of data points that match
        their actual labels
        Args:
            predicted (torch.Tensor): For each row, what's the probability that the instance belongs to each of K classes
            actual (torch.Tensor): actual class label
        NOTE: Since we have our training data in "batch form", we will be getting the result for each batch in the dataloader
        """
        prediction = torch.argmax(
            predicted, dim=-1
        )  # identify index of the most likely class

        performance = torch.where(
            (prediction == actual), torch.Tensor([1.0]), torch.Tensor([0.0])
        )  # for each row, did you predict correctly?

        return int(torch.sum(performance))

    def compute_accuracy(self, predicted, actual):
        """
        Given a prediction (usually in logit form for classification problem), identify the
        most likely label (probabilistically). Usually, for multiclass (more than 2 classes),
        Softmax is applied at the end. For binary, apply Sigmoid activation at the last
        Args:
            predicted (torch.Tensor): For each row, what's the probability that the instance belongs to each of K classes
            actual (torch.Tensor): actual class label
        NOTE: Since we have our training data in "batch form", we will be getting an accuracy for each batch in the dataloader
        """

        batch_correct_pred = self.compute_correct(predicted, actual)
        batch_accuracy = batch_correct_pred / len(predicted)

        return batch_accuracy

    def _train_init(self):
        self._train_correct = (
            0  # number of correct predictions in training set in current epoch
        )
        super()._train_init()

    def _train_step(self, inputs: torch.Tensor, labels: torch.Tensor):
        super()._train_step(inputs, labels)
        self._train_correct += self.compute_correct(self._outputs, labels)

    def _train_end(self):
        super()._train_end()
        self._train_accuracy = self._train_correct / self.epoch_train_size

    def _test_init(self):
        self._test_correct = (
            0  # number of correct predictions in testing set in current epoch
        )
        super()._test_init()

    def _test_step(self, inputs: torch.Tensor, labels: torch.Tensor):
        super()._test_step(inputs, labels)
        # currently only preserving the prediction array and label array for the last epoch for
        # confusion matrix calculation
        if self.curr_epoch == self.epochs - 1:
            self.y_pred_last_epoch.append(self._test_pred.detach().numpy().squeeze())
            self.labels_last_epoch.append(labels.detach().numpy().squeeze())
        self._test_correct += self.compute_correct(self._test_pred, labels)

    def _test_end(self):
        super()._test_end()
        self._test_accuracy = self._test_correct / self.epoch_test_size

    def _get_epoch_result(self):
        return ClassificationEpochResult(
            self.curr_epoch,
            self._mean_train_loss,
            self._mean_test_loss,
            self._epoch_time,
            self._train_accuracy,
            self._test_accuracy,
        )

    def generate_confusion_matrix(self):
        label = []
        y_pred = []

        label = np.array(self.labels_last_epoch).flatten()
        for batch in self.y_pred_last_epoch:
            y_pred = np.concatenate(
                (y_pred, np.argmax(batch, axis=1)), axis=None
            )  # flatten and concatenate
        categoryList = np.arange(0, len(self.y_pred_last_epoch[0][0])).tolist()
        return confusion_matrix(label, y_pred, labels=categoryList)

    def generate_AUC_ROC_CURVE(self) -> list[tuple[list[float], list[float], float]]:
        label_list = []
        y_preds_list = []
        plot_data = []

        # generating a numerical category list for confusion matrix axis labels, and setting up the y_preds_list and label_list for each category
        category_list = np.arange(0, len(self.y_pred_last_epoch[0][0])).tolist()
        labels_last_epoch = np.array(self.labels_last_epoch).flatten()
        label_list = np.zeros((len(self.category_list), len(labels_last_epoch)))

        for i in range(len(labels_last_epoch)):
            label_list[int(labels_last_epoch[i])][i] = 1

        y_preds_list = np.transpose(np.concatenate(np.array(self.y_pred_last_epoch)))

        for i in range(len(category_list)):
            pred_prob = np.array(y_preds_list[i])
            y_test = label_list[i]
            fpr, tpr, _ = roc_curve(y_test, pred_prob)
            auc = roc_auc_score(y_test, pred_prob)
            # this data will be sent to frontend to make interactive plotly graph
            plot_data.append((fpr.tolist(), tpr.tolist(), auc))
        return plot_data


class RegressionTrainer(Trainer[EpochResult]):
    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterionHandler: CriterionHandler,
        epochs: int,
    ):
        super().__init__(
            train_loader,
            test_loader,
            model,
            optimizer,
            criterionHandler,
            epochs,
        )
