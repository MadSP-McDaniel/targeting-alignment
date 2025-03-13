import argparse
import json
from types import SimpleNamespace
from typing import List, Optional, Union

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from clfextract.prompt_managers import HFPromptManager, PromptManager
from clfextract.utils import arange_args, filter_kwargs, type_check


class Hyperparameters:
    def __init__(self):
        """
        Base class for hyperparameters.
        """

    def __repr__(self) -> str:
        """
        Returns a string representation of the hyperparameters.

        Returns:
            str: A string representation of the hyperparameters.
        """
        attributes = [f"{attr}: {value}" for attr, value in self.__dict__.items()]
        class_name = self.__class__.__name__
        return f"{class_name}({', '.join(attributes)})"

    def __str__(self) -> str:
        """
        Returns a string representation of the hyperparameters.

        Returns:
            str: A string representation of the hyperparameters.
        """
        attributes = [f"{attr}: {value}" for attr, value in self.__dict__.items()]
        class_name = self.__class__.__name__
        return f"{class_name}\n\n" + "\n".join(attributes)


class ThreatModelConfig(Hyperparameters):
    @type_check
    def __init__(
        self,
        *,
        conv_template: Optional[str] = None,
        append_strat: Optional[str] = None,
        constraint: Optional[Union[List[int], List[str]]] = None,
        context: Optional[List[dict]] = None,
        system: Optional[str] = None,
    ):

        self.constraint = constraint
        self.conv_template = conv_template
        self.append_strat = append_strat
        self.system = system
        self.context = context

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        return super().__str__()


class ExpConfig(Hyperparameters):
    @type_check
    def __init__(
        self,
        model_path: str,
        *,
        dataset_path: Optional[str] = None,
        output_path: Optional[str] = None,
        model_device: Optional[str] = None,
        tokenizer_kwargs: dict = {},
        model_kwargs: dict = {},
        start: Optional[int] = None,
        end: Optional[int] = None,
        judge_path: Optional[str] = None,
        judge_device: Optional[str] = None,
        judge_template: Optional[str] = None,
        half: bool = False,
        no_model: bool = False,
        no_tokenizer: bool = False,
    ):
        self.model_path = model_path

        self.dataset_path = "" if dataset_path is None else dataset_path
        self.tokenizer_path = model_path

        model_device = "auto" if model_device is None else model_device
        judge_device = "auto" if judge_device is None else judge_device

        # Load the model and tokenizer
        torch_dtype = torch.float16 if half else torch.float32
        if not no_tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, **tokenizer_kwargs
            )

        else:
            self.tokenizer = None

        if model_device != "auto":
            model_device = torch.device(model_device)
        if not no_model:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                output_hidden_states=True,
                torch_dtype=torch_dtype,
                device_map=model_device,
                **model_kwargs,
            )
            if self.tokenizer.pad_token is None:
                print(
                    "No pad token found. Adding [PAD] as a pad_token and resizing the token_embedding space."
                )
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.model = None

        # Load the judge model and tokenizer if provided
        if judge_path is not None:
            if judge_device != "auto":
                judge_device = torch.device(judge_device)

            self.judge_tokenizer = AutoTokenizer.from_pretrained(judge_path)
            self.judge_model = AutoModelForCausalLM.from_pretrained(
                judge_path,
                torch_dtype=torch_dtype,
                device_map=judge_device,
            )
            self.judge_model.eval()
            self.judge_template = judge_template
        else:
            self.judge_model = None
            self.judge_tokenizer = None
            self.judge_template = None

        # Load the dataset
        # Check the extension of the dataset_path
        extension = self.dataset_path.split(".")[-1]
        match extension:
            case "csv":
                with open(dataset_path, "r") as f:
                    self.dataset = pd.read_csv(f)
            case "json":
                with open(dataset_path, "r") as f:
                    self.dataset = pd.DataFrame(json.load(f))
            case "":
                self.dataset = None
            case _:
                raise ValueError(
                    f"Invalid dataset format. It must be a csv or json file ({extension} was provided)"
                )

        if self.dataset is None:
            self.start = None
            self.end = None
        else:
            self.start = max(start, 0) if start is not None else 0
            self.end = (
                min(end, len(self.dataset)) if end is not None else len(self.dataset)
            )
            assert (
                self.start < self.end
            ), f"Invalid values for start and end. The start index must be less than the end index ({self.start} and {self.end} were provided)"

        self.output_path = output_path
        self.half = half
        self.model_device = model_device
        self.judge_path = judge_path
        self.judge_device = judge_device
        self.judge_template = judge_template
        self.no_model = no_model
        self.no_tokenizer = no_tokenizer


class GeneralConfig:
    @type_check
    def __init__(
        self,
        threat_model: ThreatModelConfig,
        exp: ExpConfig,
        misc: SimpleNamespace,
    ):
        self.threat_model = threat_model
        self.exp = exp
        self.misc = misc

        if self.threat_model.conv_template is not None:
            self.prompt_manager = PromptManager(
                self.exp.tokenizer,
                self.threat_model.conv_template,
                system=self.threat_model.system,
                append_strat=self.threat_model.append_strat,
            )
        else:
            self.prompt_manager = HFPromptManager(
                self.exp.tokenizer,
                system=self.threat_model.system,
                append_strat=self.threat_model.append_strat,
            )

    def __repr__(self) -> str:
        return f"GeneralConfig({self.threat_model}, {self.exp}, {self.misc})"

    def __str__(self) -> str:
        return f"GeneralConfig\n\n{self.threat_model}\n\n{self.exp}\n\n{self.misc}"


def set_config() -> GeneralConfig:
    """
    Parses command-line arguments to set various configurations for the application.

    Returns:
        GeneralConfig: An object containing the experiment, threat model, and miscellanous (unspecified) configurations.

    Command-line Arguments:
        --model_path (str): Model name. Default is "hf-internal-testing/tiny-random-LlamaForCausalLM".
        --dataset_path (str): Path to prompt dataset. Default is None.
        --half (bool): Use half precision.
        --start (int): Start index. Default is None.
        --end (int): End index. Default is None.
        --output_path (str): Output path. Default is None.
        --offline (bool): Use offline evaluation.
        --model_device (str): Device for model. Default is None.
        --judge_path (str): Path to judge model. Default is None.
        --judge_device (str): Device for judge model. Default is None.
        --judge_template (str): Judge conversation template. Default is None.
        --no_model (bool): Prevent loading model.
        --no_tokenizer (bool): Prevent loading tokenizer.
        --conv_template (str): Conversation template. Default is None.
        --constraint (int): Constraint. Default is None.
        --context (str): Context prompt. Default is None.
        --system (str): System prompt. Default is None.

    The function also handles any unknown arguments by storing them in a miscellaneous
    configuration namespace.
    """
    parser = argparse.ArgumentParser(description="Set hyperparameters")
    # Exp Config
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="Model name",
        default="hf-internal-testing/tiny-random-LlamaForCausalLM",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        help="Path to prompt dataset",
        default=None,
    )
    parser.add_argument("--half", action="store_true", help="Use half precision")
    parser.add_argument(
        "--start", type=int, required=False, help="Start index", default=None
    )
    parser.add_argument(
        "--end", type=int, required=False, help="End index", default=None
    )
    parser.add_argument(
        "--output_path", type=str, required=False, help="Output path", default=None
    )
    parser.add_argument(
        "--model_device",
        type=str,
        required=False,
        help="Device for model",
        default=None,
    )
    parser.add_argument(
        "--judge_path",
        type=str,
        required=False,
        help="Path to judge model",
        default=None,
    )
    parser.add_argument(
        "--judge_device",
        type=str,
        required=False,
        help="Device for judge model",
        default=None,
    )
    parser.add_argument(
        "--judge_template",
        type=str,
        required=False,
        help="Judge conversation template",
        default=None,
    )
    parser.add_argument("--no_model", action="store_true", help="Prevent loading model")
    parser.add_argument(
        "--no_tokenizer", action="store_true", help="Prevent loading tokenizer"
    )

    # Threat Model Config
    parser.add_argument(
        "--append_strat", type=str, required=False, help="Append strategy", default=None
    )
    parser.add_argument(
        "--conv_template",
        type=str,
        required=False,
        help="Conversation template",
        default=None,
    )
    parser.add_argument(
        "--constraint",
        type=int,
        nargs="+",
        required=False,
        help="Constraint",
        default=None,
    )
    parser.add_argument(
        "--context", type=str, required=False, help="Context prompt", default=None
    )
    parser.add_argument(
        "--system", type=str, required=False, help="System prompt", default=None
    )

    known_args, unknown_args = parser.parse_known_args()
    args_and_kwargs = vars(known_args)

    misc_config = {}

    i = 0
    while i < len(unknown_args):
        key = unknown_args[i].lstrip("--").lstrip("-")
        if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
            misc_config[key] = unknown_args[i + 1]
            i += 2
        else:
            misc_config[key] = True
            i += 1

    misc_config = SimpleNamespace(**misc_config)

    threat_model_config = ThreatModelConfig(
        *arange_args(ThreatModelConfig, args_and_kwargs),
        **filter_kwargs(ThreatModelConfig, args_and_kwargs),
    )

    exp_config = ExpConfig(
        *arange_args(ExpConfig, args_and_kwargs),
        **filter_kwargs(ExpConfig, args_and_kwargs),
    )

    config = GeneralConfig(
        threat_model=threat_model_config,
        exp=exp_config,
        misc=misc_config,
    )

    return config
