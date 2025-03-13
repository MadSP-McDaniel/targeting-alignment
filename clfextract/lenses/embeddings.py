from typing import List, Optional, Union

import torch

from clfextract.lenses.lenses import Lens
from clfextract.utils import find_executable_batch_size, type_check


class EmbeddingLens(Lens):
    """
    Embedding class represents a lens for embeddings in general.

    Args:
        model (Model): The model used for generating embeddings.
        tokenizer (Tokenizer): The tokenizer used for tokenizing input.
        layers (Union[int, List[int]]): The layers of the model to extract embeddings from.
        style (str): The style of embeddings to extract.

    Attributes:
        model (Model): The model used for generating embeddings.
        tokenizer (Tokenizer): The tokenizer used for tokenizing input.
        layers (Union[int, List[int]]): The layers of the model to extract embeddings from.
        style (str): The style of embeddings to extract.

    Methods:
        __call__(self, input): Computes the last embeddings for the given input.
        distance(self, input_vec, target, p): Computes the distance between input and target embeddings.

    """

    def __init__(
        self,
        model,
        tokenizer,
        layers: Optional[Union[int, List[int]]] = None,
        positions: Optional[Union[int, List[int]]] = None,
        requires_grad_: bool = False,
    ):
        super().__init__(model, tokenizer)
        self.layers = [layers] if isinstance(layers, int) else layers
        self.positions = [positions] if isinstance(positions, int) else positions

        if self.layers is None:
            print("Warning: No layers specified. Defaulting to all layers.")
            self.layers = [l for l in range(model.config.num_hidden_layers + 1)]

        if self.positions is None:
            print("Warning: No positions specified. Defaulting to all positions.")

        self.target = None
        self.requires_grad_ = requires_grad_

        assert (
            min(self.layers) >= -model.config.num_hidden_layers
            and max(self.layers) <= model.config.num_hidden_layers
        ), "Invalid layer(s) specified"

    @find_executable_batch_size
    def __call__(
        batch_size,
        self,
        input: Optional[Union[str, List[str]]] = None,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        unpad: bool = False,
    ) -> torch.Tensor:
        """
        Computes the context for the given input.

        Args:
            batch_size (int): The batch size for processing the input.
            input (Union[str, List[str], torch.Tensor]): The input text or batch of texts.

        Returns:
            torch.Tensor: The computed contextual embeddings.

        """
        assert (
            input is not None or inputs_embeds is not None or input_ids is not None
        ), "Input tensor or embeddings must be provided"

        if input is not None:
            input = [input] if isinstance(input, str) else input
            inputs = self.tokenizer(input, return_tensors="pt", padding=True).to(
                self.model.device
            )
            # Use attention mask to get the prompt length
            lengths = torch.sum(inputs.attention_mask, dim=1).cpu().numpy()
            num_inputs = inputs.input_ids.shape[0]
            max_length = inputs.input_ids.shape[1]

        elif input_ids is not None or inputs_embeds is not None:
            inputs = input_ids if input_ids is not None else inputs_embeds
            lengths = None
            num_inputs = inputs.shape[0]
            max_length = inputs.shape[1]

        positions = (
            [i for i in range(max_length)] if self.positions is None else self.positions
        )

        emb = torch.zeros(
            (
                num_inputs,
                len(self.layers),
                len(positions),
                self.model.config.hidden_size,
            ),
            device=self.model.device,
        )

        for i in range(0, num_inputs, batch_size):
            if input is not None:
                batch_inputs = {
                    k: v[i : min(i + batch_size, num_inputs)] for k, v in inputs.items()
                }
                output = self.model(**batch_inputs)

            elif input_ids is not None:
                batch_inputs = inputs[i : min(i + batch_size, num_inputs), :]
                output = self.model(
                    input_ids=batch_inputs, attention_mask=attention_mask
                )

            elif inputs_embeds is not None:
                batch_inputs = inputs[i : min(i + batch_size, num_inputs), :, :]
                output = self.model(
                    inputs_embeds=batch_inputs, attention_mask=attention_mask
                )

            hidden_states = output.hidden_states

            for j, layer in enumerate(self.layers):
                emb[i : min(i + batch_size, num_inputs), j, :] = (
                    hidden_states[layer][:, :]
                    if self.positions is None
                    else hidden_states[layer][:, positions, :]
                )

            # Delete hidden_states to free up memory
            del hidden_states
            torch.cuda.empty_cache()

        if self.requires_grad_:
            emb.requires_grad_()

        if return_numpy:
            emb = emb.cpu().detach().numpy()

        if self.positions is None and lengths is not None and return_numpy and unpad:
            emb = [emb[i, :, -lengths[i] :, :] for i in range(emb.shape[0])]

        return emb

    def distance(
        self, input_vec, target=None, p: Optional[Union[int, str]] = 2
    ) -> float:
        """
        Computes the distance between input and target embeddings.

        Args:
            input_vec (torch.Tensor): The input embeddings.
            target (torch.Tensor): The target embeddings.
            p (Union[int, str], optional): The type of distance lens to use. Defaults to 2.

        Returns:
            float: The computed distance between input and target embeddings.

        Raises:
            ValueError: If the dimensions of input and target embeddings do not match.
            ValueError: If an invalid type of distance lens is provided.

        """
        # Remove dimensions of size 1
        input_vec = input_vec.squeeze()
        target = target if target is not None else self.target
        target = target.squeeze()
        dim_axis = (
            1 if len(input_vec.shape) == 2 else 0
        )  # We assume that the batch size, if not 1, corresponds to the first axis

        # Check that dimensions match
        if input_vec.shape != target.shape:
            raise ValueError(
                f"input and target dimensions do not match. Input has shape {input_vec.shape} and target has shape {target.shape}"
            )

        # Compute the distance between the input and target
        if p == 1:
            result = torch.norm(input_vec - target, p=1, dim=dim_axis)
        elif p == 2:
            result = torch.norm(input_vec - target, p=2, dim=dim_axis)
        elif p == "inf":
            result = torch.norm(input_vec - target, p=float("inf"), dim=dim_axis)
        elif p == "cosine":
            result = torch.dot(input_vec, target) / (
                torch.norm(input_vec) * torch.norm(target) + 1e-8
            )
        else:
            raise ValueError(f"Invalid type of distance. (provided p={p})")

        if not self.requires_grad_:
            result = result.item()

        return result


class KVLens(Lens):
    @type_check
    def __init__(
        self,
        model,
        tokenizer,
        type: str,  # "key", "value"
        layers: Optional[Union[int, List[int]]] = None,
        heads: Optional[Union[int, List[int]]] = None,
        positions: Optional[Union[int, List[int]]] = None,
        requires_grad_: bool = False,
    ):
        super().__init__(model, tokenizer)
        assert type in [
            "key",
            "value",
        ], "Invalid type specified. Must be either 'key' or 'value'"
        self.layers = [layers] if isinstance(layers, int) else layers
        self.positions = [positions] if isinstance(positions, int) else positions
        self.heads = [heads] if isinstance(heads, int) else heads

        self.embed_size = (
            self.model.config.hidden_size // self.model.config.num_attention_heads
        )

        if self.layers is None:
            print("Warning: No layers specified. Default to all layers.")
            self.layers = [
                l for l in range(model.config.num_hidden_layers)
            ]  # num_hidden_layers also considers first embedding conversion

        if self.heads is None:
            print("Warning: No heads specified. Default to all heads.")
            self.heads = [h for h in range(model.config.num_attention_heads)]

        if self.positions is None:
            print("Warning: No positions specified. Default to all positions.")

        self.target = None
        self.requires_grad_ = requires_grad_
        self.type = 0 if type == "key" else 1

        assert (
            min(self.layers) >= -model.config.num_hidden_layers
            and max(self.layers) <= model.config.num_hidden_layers
        ), "Invalid layer(s) specified"

    @find_executable_batch_size
    def __call__(
        batch_size,
        self,
        input: Union[str, List[str], torch.Tensor],
        return_numpy: bool = False,
        unpad: bool = False,
    ) -> torch.Tensor:
        """
        Computes the context for the given input.

        Args:
            batch_size (int): The batch size for processing the input.
            input (Union[str, List[str], torch.Tensor]): The input text or batch of texts.

        Returns:
            torch.Tensor: The computed contextual embeddings.

        """
        assert (
            isinstance(input, str)
            or isinstance(input, list)
            or isinstance(input, torch.Tensor)
        ), "Input must be a string, a list of strings, or a tensor"

        if isinstance(input, torch.Tensor):
            inputs = input
            lengths = None
        else:
            input = [input] if isinstance(input, str) else input
            inputs = self.tokenizer(input, return_tensors="pt", padding=True).to(
                self.model.device
            )
            # Use attention mask to get the prompt length
            lengths = torch.sum(inputs.attention_mask, dim=1).cpu().numpy()

        positions = (
            [i for i in range(inputs.input_ids.shape[1])]
            if self.positions is None
            else self.positions
        )

        emb = torch.zeros(
            (
                inputs.input_ids.shape[0],
                len(self.layers),
                len(self.heads),
                len(positions),
                self.embed_size,
            ),
            device=self.model.device,
        )

        for i in range(0, inputs.input_ids.shape[0], batch_size):
            batch_inputs = {
                k: v[i : min(i + batch_size, inputs.input_ids.shape[0])]
                for k, v in inputs.items()
            }
            output = self.model(**batch_inputs)
            past_key_values = output.past_key_values

            for j, layer in enumerate(self.layers):
                emb[i : min(i + batch_size, inputs.input_ids.shape[0]), j] = (
                    past_key_values[layer][self.type][:, self.heads, positions, :]
                    if self.positions is not None
                    else past_key_values[layer][self.type][:, self.heads, :, :]
                )

        if self.requires_grad_:
            emb.requires_grad_()

        if return_numpy:
            emb = emb.cpu().detach().numpy()

        if self.positions is None and lengths is not None and return_numpy and unpad:
            emb = [emb[i, :, -lengths[i] :, :] for i in range(emb.shape[0])]

        return emb

    def distance(self) -> float:
        raise NotImplementedError("Distance not implemented for KVLens")


def pruned_forward(
    model,
    layer_ids: List[int],
    input_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    keep_all_layers: bool = True,
    keep_conversion_layer: bool = True,
) -> List[torch.Tensor]:
    """
    Forward pass through the model with pruned layers.
    Args:
        model (Model): The model to forward pass through.
        input_ids (torch.Tensor): The input tensor.
        layers (List[int]): The layers to forward pass through.
        position_ids (torch.Tensor): The position IDs for the input tensor.
    Returns:
        List[torch.Tensor]: The hidden states from the specified layers.
    """
    assert isinstance(input_ids, torch.Tensor) or isinstance(
        inputs_embeds, torch.Tensor
    ), "Input tensor or embeddings must be provided"
    assert (
        min(layer_ids) >= 0 and max(layer_ids) < model.config.num_hidden_layers
    ), "Invalid layer(s) specified"
    assert len(layer_ids) > 0, "No layers specified"
    assert (
        isinstance(position_ids, torch.Tensor) or position_ids is None
    ), "Position IDs must be a tensor or None"
    layer_ids.sort()
    # Get the embeddings
    if input_ids is not None:
        inputs_embeds = model.embed_tokens(
            input_ids
        )  # TODO : Generalize to all models?
        # Prepare attention mask
        attention_mask = torch.ones_like(input_ids)
        bsz, seq_len = input_ids.size()
        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            .unsqueeze(0)
            .expand(bsz, -1)
            if position_ids is None
            else position_ids
        )

    elif inputs_embeds is not None and input_ids is None:
        attention_mask = torch.ones(
            inputs_embeds.size()[:2], device=inputs_embeds.device, dtype=torch.int64
        )
        bsz, seq_len, _ = inputs_embeds.size()
        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=inputs_embeds.device)
            .unsqueeze(0)
            .expand(bsz, -1)
            if position_ids is None
            else position_ids
        )

    # Prepare causal mask
    causal_mask = model._update_causal_mask(
        attention_mask,
        inputs_embeds,
        torch.arange(seq_len, device=inputs_embeds.device),
        None,  # past_key_values
        False,  # output_attentions
    )

    layer_kwargs = {
        "attention_mask": causal_mask,
        "position_ids": position_ids,
        "past_key_value": None,
        "output_attentions": False,
        "use_cache": False,
        "cache_position": None,
    }

    # Generate position embeddings
    if hasattr(model, "rotary_emb") and model.rotary_emb is not None:
        position_embeddings = model.rotary_emb(
            inputs_embeds, position_ids
        )  # Llama specific
        layer_kwargs["position_embeddings"] = position_embeddings
    # Start with the input embeddings as hidden states
    hidden_states = inputs_embeds
    # List to store all hidden states
    all_hidden_states = (
        [hidden_states] if keep_all_layers and keep_conversion_layer else []
    )
    # Forward pass through specified layers

    for layer in [model.layers[l] for l in layer_ids]:
        layer_outputs = layer(
            hidden_states,
            **layer_kwargs,
        )
        hidden_states = layer_outputs[0]
        if keep_all_layers:
            all_hidden_states.append(hidden_states)
        elif not keep_all_layers and model.layers[-1] == layer:
            all_hidden_states.append(hidden_states)

    # Final layer norm if the last layer is not pruned
    if layer_ids[-1] == len(model.layers) - 1 and model.norm is not None:
        hidden_states = model.norm(hidden_states)
        all_hidden_states[-1] = hidden_states

    return all_hidden_states


def truncated_forward(
    model,
    start_layer: int,
    end_layer: Optional[int] = None,
    input_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    keep_all_layers: bool = True,
    keep_conversion_layer: bool = True,
):
    """
    Forward pass through the model from a specified layer to another specified layer.
    Args:
        model (Model): The model to forward pass through.
        input_ids (torch.Tensor): The input tensor.
        start_layer (int): The layer to start from.
        end_layer (int): The layer to end at.
        position_ids (torch.Tensor): The position IDs for the input tensor.
    Returns:
        List[torch.Tensor]: The hidden states from the specified layers.
    """
    assert isinstance(input_ids, torch.Tensor) or isinstance(
        inputs_embeds, torch.Tensor
    ), "Input tensor or embeddings must be provided"
    assert (
        start_layer >= 0 and start_layer < model.config.num_hidden_layers
    ), f"Invalid start_layer, must be between 0 and {model.config.num_hidden_layers} ({start_layer} was provided)"
    assert end_layer is None or (
        start_layer <= end_layer and end_layer < model.config.num_hidden_layers
    ), f"Invalid end_layer, must be between {start_layer} and {model.config.num_hidden_layers-1} ({end_layer} was provided)"

    end_layer = model.config.num_hidden_layers - 1 if end_layer is None else end_layer
    layer_ids = [l for l in range(start_layer, end_layer + 1)]

    if input_ids is not None:
        return pruned_forward(
            model,
            layer_ids,
            input_ids=input_ids,
            position_ids=position_ids,
            keep_all_layers=keep_all_layers,
            keep_conversion_layer=keep_conversion_layer,
        )
    else:
        return pruned_forward(
            model,
            layer_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            keep_all_layers=keep_all_layers,
            keep_conversion_layer=keep_conversion_layer,
        )


class PrunedEmbeddingLens(EmbeddingLens):
    @type_check
    def __init__(
        self,
        model,
        tokenizer,
        layers: Optional[List[int]] = None,
        keep_conversion_layer: bool = True,
        **kwargs,
    ):

        assert (
            min(layers) >= 0 and max(layers) < model.config.num_hidden_layers
        ), "Invalid layer(s) specified"
        if layers is None:
            print("Warning: No layers specified. Defaulting to all layers.")
            layers = [l for l in range(model.config.num_hidden_layers)]

        super().__init__(
            model,
            tokenizer,
            layers=layers,
            **kwargs,
        )

        self.keep_conversion_layer = keep_conversion_layer

    @find_executable_batch_size
    def __call__(
        batch_size,
        self,
        input: Union[str, List[str], torch.Tensor],
        return_numpy: bool = False,
        unpad: bool = False,
    ) -> torch.Tensor:
        """
        Computes the context for the given input.

        Args:
            batch_size (int): The batch size for processing the input.
            input (Union[str, List[str], torch.Tensor]): The input text or batch of texts.

        Returns:
            torch.Tensor: The computed contextual embeddings.

        """
        assert (
            isinstance(input, str)
            or isinstance(input, list)
            or isinstance(input, torch.Tensor)
        ), "Input must be a string, a list of strings, or a tensor"

        if isinstance(input, torch.Tensor):
            inputs = input
            lengths = None
        else:
            input = [input] if isinstance(input, str) else input
            inputs = self.tokenizer(input, return_tensors="pt", padding=True).to(
                self.model.device
            )
            # Use attention mask to get the prompt length
            lengths = torch.sum(inputs.attention_mask, dim=1).cpu().numpy()

        positions = (
            [i for i in range(inputs.input_ids.shape[1])]
            if self.positions is None
            else self.positions
        )

        emb = torch.zeros(
            (
                inputs.input_ids.shape[0],
                len(self.layers),
                len(positions),
                self.model.config.hidden_size,
            ),
            device=self.model.device,
        )

        for i in range(0, inputs.input_ids.shape[0], batch_size):
            batch_inputs = {
                k: v[i : min(i + batch_size, inputs.input_ids.shape[0])]
                for k, v in inputs.items()
            }
            hidden_states = pruned_forward(
                self.model.model,
                self.layers,
                input_ids=batch_inputs["input_ids"],
                keep_conversion_layer=self.keep_conversion_layer,
            )

            for j in range(len(hidden_states)):
                emb[i : min(i + batch_size, inputs.input_ids.shape[0]), j, :] = (
                    hidden_states[j][:, :]
                    if self.positions is None
                    else hidden_states[j][:, self.positions, :]
                )

        if self.requires_grad_:
            emb.requires_grad_()

        if return_numpy:
            emb = emb.cpu().detach().numpy()

        if self.positions is None and lengths is not None and return_numpy and unpad:
            emb = [emb[i, :, -lengths[i] :, :] for i in range(emb.shape[0])]

        return emb


class TruncatedEmbeddingLens(EmbeddingLens):
    @type_check
    def __call__(
        batch_size,
        self,
        input: Union[str, List[str], torch.Tensor],
        return_numpy: bool = False,
        unpad: bool = False,
    ) -> torch.Tensor:
        """
        Computes the context for the given input.

        Args:
            batch_size (int): The batch size for processing the input.
            input (Union[str, List[str], torch.Tensor]): The input text or batch of texts.

        Returns:
            torch.Tensor: The computed contextual embeddings.

        """
        assert (
            isinstance(input, str)
            or isinstance(input, list)
            or isinstance(input, torch.Tensor)
        ), "Input must be a string, a list of strings, or a tensor"

        if isinstance(input, torch.Tensor):
            inputs = input
            lengths = None
        else:
            input = [input] if isinstance(input, str) else input
            inputs = self.tokenizer(input, return_tensors="pt", padding=True).to(
                self.model.device
            )
            # Use attention mask to get the prompt length
            lengths = torch.sum(inputs.attention_mask, dim=1).cpu().numpy()

        positions = (
            [i for i in range(inputs.input_ids.shape[1])]
            if self.positions is None
            else self.positions
        )

        emb = torch.zeros(
            (
                inputs.input_ids.shape[0],
                len(self.layers),
                len(positions),
                self.model.config.hidden_size,
            ),
            device=self.model.device,
        )

        for i in range(0, inputs.input_ids.shape[0], batch_size):
            batch_inputs = {
                k: v[i : min(i + batch_size, inputs.input_ids.shape[0])]
                for k, v in inputs.items()
            }
            hidden_states = truncated_forward(
                self.model.model,
                self.start_layer,
                self.end_layer,
                input_ids=batch_inputs["input_ids"],
                keep_conversion_layer=self.keep_conversion_layer,
            )

            for j in range(len(hidden_states)):
                emb[i : min(i + batch_size, inputs.input_ids.shape[0]), j, :] = (
                    hidden_states[j][:, :]
                    if self.positions is None
                    else hidden_states[j][:, self.positions, :]
                )

            # Delete hidden_states to free up memory
            del hidden_states
            torch.cuda.empty_cache()

        if self.requires_grad_:
            emb.requires_grad_()

        if return_numpy:
            emb = emb.cpu().detach().numpy()

        if self.positions is None and lengths is not None and return_numpy and unpad:
            emb = [emb[i, :, -lengths[i] :, :] for i in range(emb.shape[0])]

        return emb
