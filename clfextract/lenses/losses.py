import torch.nn as nn

from clfextract.lenses.lenses import Lens
from clfextract.prompt_managers import PromptManager


class LossLens(Lens):
    """
    A class representing a loss lens.

    Inherits from Scout class.
    """

    def __init__(self, model, tokenizer, prompt_manager: PromptManager) -> None:
        """
        Initializes a Loss object.

        Args:
            model (str): The model.
            tokenizer (str): The tokenizer.
            prompt_manager (PromptManager): The prompt manager.
        """
        super().__init__(model, tokenizer)
        self.prompt_manager = prompt_manager

    def __call__(self, attack: str = "") -> float:
        raise NotImplementedError("Default Loss class does not have a __call__ method")


class PerplexityLossLens(LossLens):
    """
    A class representing a target loss lens.

    Inherits from Loss class.
    """

    def __init__(self, model, tokenizer, prompt_manager: PromptManager) -> None:
        """
        Initializes a TargetLoss object.

        Args:
            model (str): The model used by the loss scout.
            tokenizer (str): The tokenizer used by the loss scout.
        """
        super().__init__(model, tokenizer, prompt_manager)

    def __call__(self, attack: str = "") -> float:
        input_ids = self.prompt_manager.get_input_ids(attack)
        loss = self.model(input_ids, labels=input_ids.clone()).loss
        return loss.item()


class TargetLossLens(LossLens):
    """
    A class representing a target loss lens.

    Inherits from Loss class.
    """

    def __init__(self, model, tokenizer, prompt_manager) -> None:
        """
        Initializes a TargetLoss object.

        Args:
            model (str): The model used by the loss scout.
            tokenizer (str): The tokenizer used by the loss scout.
        """
        super().__init__(model, tokenizer, prompt_manager)

    def __call__(self, attack: str = "") -> float:
        input_ids = self.prompt_manager.get_input_ids(attack)
        logits = self.prompt_manager.get_logits(attack)
        loss_slice = self.prompt_manager._loss_slice
        target_slice = self.prompt_manager._target_slice
        crit = nn.CrossEntropyLoss(reduction="none")
        loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
        loss = crit(
            logits[:, loss_slice, :].transpose(1, 2), input_ids[:, target_slice]
        )
        return loss.mean(dim=-1)
