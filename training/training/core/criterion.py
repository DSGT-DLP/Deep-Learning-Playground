from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class CriterionHandler(ABC):
    def compute_loss(self, output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        postprocess_output = output.clone()
        postprocess_labels = labels.clone()
        return self._compute_loss(postprocess_output, postprocess_labels)

    @abstractmethod
    def _compute_loss(self, output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pass


class L1LossHandler(CriterionHandler):
    def _compute_loss(self, output, labels):
        output = torch.reshape(
            output,
            (output.shape[0], output.shape[2]),
        )
        return nn.L1Loss()(output, labels)


class MSELossHandler(CriterionHandler):
    def _compute_loss(self, output, labels):
        output = torch.reshape(
            output,
            (output.shape[0], output.shape[2]),
        )
        return nn.MSELoss()(output, labels)


class BCELossHandler(CriterionHandler):
    def _compute_loss(self, output, labels):
        return nn.BCELoss()(output, labels.unsqueeze(1))


class BCEWithLogitsLossHandler(CriterionHandler):
    def _compute_loss(self, output, labels):
        return nn.BCEWithLogitsLoss()(output, labels.unsqueeze(1))


class CELossHandler(CriterionHandler):
    def _compute_loss(self, output, labels):
        output = torch.reshape(
            output,
            (output.shape[0], output.shape[2]),
        )
        labels = labels.squeeze_()
        return nn.CrossEntropyLoss(reduction="mean")(output, labels.long())


class WCELossHandler(CriterionHandler):
    def _compute_loss(self, output, labels):
        output = torch.reshape(
            output,
            (output.shape[0], output.shape[2]),
        )
        labels = labels.squeeze_()
        return nn.CrossEntropyLoss(reduction="mean")(output, labels.long())


CRITERION_HANDLERS = {
    "L1LOSS": L1LossHandler(),
    "MSELOSS": MSELossHandler(),
    "BCELOSS": BCELossHandler(),
    "BCEWITHLOGITSLOSS": BCEWithLogitsLossHandler(),
    "CELOSS": CELossHandler(),
    "WCELOSS": WCELossHandler(),
}


def getCriterionHandler(criterion_name):
    return CRITERION_HANDLERS[criterion_name]
