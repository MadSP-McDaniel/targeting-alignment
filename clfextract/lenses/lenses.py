class Lens:
    """
    A class representing a lens for evaluating models.

    Args:
        model (str): The model being evaluated.
        tokenizer (str): The tokenizer used for the model.

    Attributes:
        model (str): The model being evaluated.
        tokenizer (str): The tokenizer used for the model.

    Methods:
        __call__(self, input): Placeholder method for computation logic.
        __str__(self): Returns a string representation of the lens.
        distance(self, input, target, type="l2"): Placeholder method for distance computation logic.
    """

    def __init__(self, model, tokenizer) -> None:
        """
        Initializes a Lens object.

        Args:
            model (str): The model being evaluated.
            tokenizer (str): The tokenizer used for the model.

        Raises:
            AssertionError: If invalid layer(s) are specified.
        """
        self.model = model
        self.tokenizer = tokenizer

    def __str__(self) -> str:
        """
        Returns a string representation of the lens.

        Returns:
            str: A string representation of the lens.
        """
        return f"{self.model} | {self.tokenizer}"
