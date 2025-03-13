from typing import List, Optional

import torch
from fastchat.conversation import get_conv_template


def load_conversation_template(template_name):
    conv_template = get_conv_template(template_name)
    if conv_template.name == "zero_shot":
        conv_template.roles = tuple(["### " + r for r in conv_template.roles])
        conv_template.sep = "\n"
    elif conv_template.name == "llama-2":
        conv_template.sep2 = conv_template.sep2.strip()

    return conv_template


# Slightly modified from https://github.com/llm-attacks/llm-attacks/blob/098262edf85f807224e70ecd87b9d83716bf6b73/llm_attacks/base/attack_manager.py#L401C27-L401C29
class PromptManager:
    def __init__(
        self,
        tokenizer,
        conv_template,
        base: str = "",
        attack: str = "",
        target: str = "",
        system: str = "",
        append_strat: Optional[str] = None,
        mark: Optional[str] = None,
    ):

        self.tokenizer = tokenizer
        self.conv_template = (
            conv_template
            if not isinstance(conv_template, str)
            else load_conversation_template(conv_template)
        )
        self.base = base
        self.target = target
        self.system = system
        self.append_strat = append_strat
        self.attack = attack
        self.mark = None
        if self.system != "" and self.system is not None:
            self.conv_template.set_system_message(self.system)

    def get_between(self, base="", attack=""):
        base = base if base != "" else self.base
        attack = attack if attack != "" else self.attack

        if attack is not None:
            self.attack = attack.replace(self.mark, base.lower())

        self.conv_template.append_message(self.conv_template.roles[0], f"{self.attack}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()

        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids

        if self.conv_template.name == "llama-2":
            self.conv_template.messages = []

            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(f"{self.attack}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._goal_slice = slice(
                self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks))
            )
            self._control_slice = self._goal_slice

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2)
            self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 3)

        else:
            python_tokenizer = False or self.conv_template.name == "oasst_pythia"
            try:
                encoding.char_to_token(len(prompt) - 1)
            except:
                python_tokenizer = True

            if python_tokenizer:
                self.conv_template.messages = []

                self.conv_template.append_message(self.conv_template.roles[0], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._user_role_slice = slice(None, len(toks))

                self.conv_template.update_last_message(f"{self.attack}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._goal_slice = slice(
                    self._user_role_slice.stop,
                    max(self._user_role_slice.stop, len(toks) - 1),
                )
                self._control_slice = self._goal_slice

                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

                self.conv_template.update_last_message(f"{self.target}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._target_slice = slice(
                    self._assistant_role_slice.stop, len(toks) - 1
                )
                self._loss_slice = slice(
                    self._assistant_role_slice.stop - 1, len(toks) - 2
                )
            else:
                self._system_slice = slice(
                    None, encoding.char_to_token(len(self.conv_template.system_message))
                )
                self._user_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
                    encoding.char_to_token(
                        prompt.find(self.conv_template.roles[0])
                        + len(self.conv_template.roles[0])
                        + 1
                    ),
                )
                self._goal_slice = slice(
                    encoding.char_to_token(prompt.find(self.attack)),
                    encoding.char_to_token(prompt.find(self.attack) + len(self.attack)),
                )
                self._control_slice = self._goal_slice
                self._assistant_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
                    encoding.char_to_token(
                        prompt.find(self.conv_template.roles[1])
                        + len(self.conv_template.roles[1])
                        + 1
                    ),
                )
                self._target_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)),
                    encoding.char_to_token(prompt.find(self.target) + len(self.target)),
                )
                self._loss_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)) - 1,
                    encoding.char_to_token(prompt.find(self.target) + len(self.target))
                    - 1,
                )

        self.conv_template.messages = []

        return prompt

    def get_prefix(self, base="", attack=""):
        base = base if base != "" else self.base
        attack = attack if attack != "" else self.attack

        if attack != "":
            self.conv_template.append_message(
                self.conv_template.roles[0], f"{attack} {base}"
            )  # Note : space seems to be added to retrieve more easily.
        else:
            self.conv_template.append_message(self.conv_template.roles[0], f"{base}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()

        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids
        self.conv_template.messages = []

        if self.conv_template.name == "llama-2":
            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(f"{attack}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._control_slice = slice(
                self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks))
            )

            separator = " " if base and attack != "" else ""
            self.conv_template.update_last_message(f"{attack}{separator}{base}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._goal_slice = slice(self._control_slice.stop, len(toks))

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._goal_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2)
            self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 3)

        else:
            python_tokenizer = False or self.conv_template.name == "oasst_pythia"
            try:
                encoding.char_to_token(len(prompt) - 1)
            except:
                python_tokenizer = True

            if python_tokenizer:
                # Update the conversation template for vicuna and pythia tokenizer and prompt
                # Append None to the user role message
                self.conv_template.append_message(self.conv_template.roles[0], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._user_role_slice = slice(None, len(toks))

                # Update the last message with the base
                self.conv_template.update_last_message(f"{attack}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._control_slice = slice(
                    self._user_role_slice.stop,
                    max(self._user_role_slice.stop, len(toks) - 1),
                )

                separator = " " if base and attack != "" else ""
                # Update the last message with the base and attack
                self.conv_template.update_last_message(f"{attack}{separator}{base}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._goal_slice = slice(self._control_slice.stop, len(toks) - 1)

                # Append None to the assistant role message
                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(self._goal_slice.stop, len(toks))

                # Update the last message with the target
                self.conv_template.update_last_message(f"{self.target}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._target_slice = slice(
                    self._assistant_role_slice.stop, len(toks) - 1
                )
                self._loss_slice = slice(
                    self._assistant_role_slice.stop - 1, len(toks) - 2
                )
            else:
                # Update the conversation template for other tokenizers and prompts

                self._system_slice = slice(
                    None, encoding.char_to_token(len(self.conv_template.system_message))
                )
                self._user_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
                    encoding.char_to_token(
                        prompt.find(self.conv_template.roles[0])
                        + len(self.conv_template.roles[0])
                        + 1
                    ),
                )
                self._control_slice = slice(
                    encoding.char_to_token(prompt.find(attack)),
                    encoding.char_to_token(prompt.find(attack) + len(attack)),
                )
                self._goal_slice = slice(
                    encoding.char_to_token(prompt.find(base)),
                    encoding.char_to_token(prompt.find(base) + len(base)),
                )
                self._assistant_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
                    encoding.char_to_token(
                        prompt.find(self.conv_template.roles[1])
                        + len(self.conv_template.roles[1])
                        + 1
                    ),
                )
                self._target_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)),
                    encoding.char_to_token(prompt.find(self.target) + len(self.target)),
                )
                self._loss_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)) - 1,
                    encoding.char_to_token(prompt.find(self.target) + len(self.target))
                    - 1,
                )

        self.conv_template.messages = []
        return prompt

    def get_suffix(self, base="", attack=""):
        base = base if base != "" else self.base
        attack = attack if attack != "" else self.attack

        if attack != "":
            self.conv_template.append_message(
                self.conv_template.roles[0], f"{base} {attack}"
            )  # Note : space seems to be added to retrieve more easily.
        else:
            self.conv_template.append_message(self.conv_template.roles[0], f"{base}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()

        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids
        self.conv_template.messages = []

        if self.conv_template.name == "llama-2":
            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(f"{base}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._goal_slice = slice(
                self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks))
            )

            separator = " " if base and attack != "" else ""
            self.conv_template.update_last_message(f"{base}{separator}{attack}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._control_slice = slice(self._goal_slice.stop, len(toks))

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2)
            self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 3)

        else:
            python_tokenizer = False or self.conv_template.name == "oasst_pythia"
            try:
                encoding.char_to_token(len(prompt) - 1)
            except:
                python_tokenizer = True

            if python_tokenizer:
                # Update the conversation template for vicuna and pythia tokenizer and prompt
                # Append None to the user role message
                self.conv_template.append_message(self.conv_template.roles[0], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._user_role_slice = slice(None, len(toks))

                # Update the last message with the base
                self.conv_template.update_last_message(f"{base}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._goal_slice = slice(
                    self._user_role_slice.stop,
                    max(self._user_role_slice.stop, len(toks) - 1),
                )

                separator = " " if base and attack != "" else ""
                # Update the last message with the base and attack
                self.conv_template.update_last_message(f"{base}{separator}{attack}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._control_slice = slice(self._goal_slice.stop, len(toks) - 1)

                # Append None to the assistant role message
                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

                # Update the last message with the target
                self.conv_template.update_last_message(f"{self.target}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._target_slice = slice(
                    self._assistant_role_slice.stop, len(toks) - 1
                )
                self._loss_slice = slice(
                    self._assistant_role_slice.stop - 1, len(toks) - 2
                )
            else:
                # Update the conversation template for other tokenizers and prompts

                self._system_slice = slice(
                    None, encoding.char_to_token(len(self.conv_template.system_message))
                )
                self._user_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
                    encoding.char_to_token(
                        prompt.find(self.conv_template.roles[0])
                        + len(self.conv_template.roles[0])
                        + 1
                    ),
                )
                self._goal_slice = slice(
                    encoding.char_to_token(prompt.find(base)),
                    encoding.char_to_token(prompt.find(base) + len(base)),
                )
                self._control_slice = slice(
                    encoding.char_to_token(prompt.find(attack)),
                    encoding.char_to_token(prompt.find(attack) + len(attack)),
                )
                self._assistant_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
                    encoding.char_to_token(
                        prompt.find(self.conv_template.roles[1])
                        + len(self.conv_template.roles[1])
                        + 1
                    ),
                )
                self._target_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)),
                    encoding.char_to_token(prompt.find(self.target) + len(self.target)),
                )
                self._loss_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)) - 1,
                    encoding.char_to_token(prompt.find(self.target) + len(self.target))
                    - 1,
                )

        self.conv_template.messages = []

        return prompt

    def get_prompt(self, base="", attack=""):
        match self.append_strat:
            case "prefix":
                prompt = self.get_prefix(base=base, attack=attack)
            case "suffix":
                prompt = self.get_suffix(base=base, attack=attack)
            case "between":
                prompt = self.get_between(base=base, attack=attack)
            case None:
                prompt = self.get_suffix(base=base, attack="")
            case _:
                raise ValueError(f"Invalid append_strat: {self.append_strat}")

        return prompt

    def get_batch_prompt(self, attacks=[], bases=None):
        assert bases is None or len(bases) == len(
            attacks
        ), "The number of bases and attacks must be the same."
        if bases is None:
            prompts = [self.get_prompt(attack=attack) for attack in attacks]
        else:
            prompts = []
            original_base = self.base
            for i in range(len(bases)):
                self.base = bases[i]
                prompts.append(self.get_prompt(attack=attacks[i]))
            self.base = original_base
        return prompts

    def get_input_ids(self, base="", attack=""):
        # Get the input_ids tensor for the conversation prompt

        prompt = self.get_prompt(base=base, attack=attack)
        toks = self.tokenizer(prompt).input_ids
        input_ids = (
            torch.tensor(toks[: self._target_slice.stop])
            if self.target != ""
            else torch.tensor(toks)
        )

        return input_ids

    def get_batch_input_ids(self, attacks=[], bases=None):
        assert bases is None or len(bases) == len(
            attacks
        ), "The number of bases and attacks must be the same."
        if bases is None:
            input_ids = [self.get_input_ids(attack=attack) for attack in attacks]
        else:
            input_ids = []
            original_base = self.base
            for i in range(len(bases)):
                self.base = bases[i]
                input_ids.append(self.get_input_ids(attack=attacks[i]))
            self.base = original_base
        return input_ids

    def get_logits(self, model, base="", attack=""):
        # Get the logits tensor for the conversation prompt

        input_ids = self.get_input_ids(base=base, attack=attack)
        logits = model(input_ids.unsqueeze(0))[0]

        return logits

    def get_slices(self):
        return {
            "user_role": self._user_role_slice,
            "goal": self._goal_slice,
            "control": self._control_slice,
            "assistant_role": self._assistant_role_slice,
            "target": self._target_slice,
            "loss": self._loss_slice,
        }


class HFPromptManager:
    """
    Class to manage the prompt for the Hugging Face models assuming tokenizer.apply_chat_template exists.
    """

    def __init__(
        self,
        tokenizer,
        base: str = "",
        attack: str = "",
        target: str = "",
        system: str = "",
        append_strat: Optional[str] = None,
        add_space: bool = False,
    ):
        assert (
            hasattr(tokenizer, "apply_chat_template")
            and tokenizer.chat_template is not None
        ), "The tokenizer must have the apply_chat_template method."
        self.tokenizer = tokenizer
        self.base = base
        self.target = target
        self.system = system
        self.append_strat = append_strat
        self.attack = attack
        self.add_space = add_space

    def get_prompt(
        self, base="", attack="", add_generation_prompt=True, append_strat=None
    ):
        messages = []
        if self.system != "" and self.system is not None:
            messages.append({"role": "system", "content": self.system})

        base = base if base != "" else self.base
        attack = attack if attack != "" else self.attack
        append_strat = append_strat if append_strat is not None else self.append_strat
        separator = " " if self.add_space and base and attack != "" else ""

        match append_strat:
            case "prefix":
                content = attack + separator + base
            case "suffix":
                content = base + separator + attack
            case None:
                content = base
            case _:
                raise ValueError(f"Invalid append_strat: {self.append_strat}")

        messages.append({"role": "user", "content": content})

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

        return prompt

    def get_batch_prompt(
        self, bases: List[str] = [], attacks: Optional[List[str]] = None
    ):
        assert attacks is None or len(bases) == len(
            attacks
        ), "The number of bases and attacks must be the same."
        if attacks is None:
            prompts = [self.get_prompt(base=base) for base in bases]
        else:
            prompts = []
            original_base = self.base
            for i in range(len(bases)):
                self.base = bases[i]
                prompts.append(self.get_prompt(attack=attacks[i]))
            self.base = original_base
        return prompts

    def get_input_ids(
        self,
        base: str = "",
        attack: str = "",
        add_generation_prompt: bool = True,
        return_tensors: str = "pt",
        **kwargs,
    ):
        # Get the input_ids tensor for the conversation prompt
        prompt = self.get_prompt(
            attack=attack, base=base, add_generation_prompt=add_generation_prompt
        )

        input_ids = self.tokenizer(
            prompt, return_tensors=return_tensors, **kwargs
        ).input_ids

        return input_ids

    def get_batch_input_ids(
        self,
        bases: List[str] = [],
        attacks: Optional[List[str]] = None,
        return_tensors: str = "pt",
        **kwargs,
    ):
        assert attacks is None or len(bases) == len(
            attacks
        ), "The number of bases and attacks must be the same."
        if attacks is None:
            attacks = ["" for _ in range(len(bases))]
        prompts = self.get_batch_prompt(attacks=attacks, bases=bases)
        input_ids = [
            self.tokenizer(prompt, return_tensors=return_tensors, **kwargs).input_ids
            for prompt in prompts
        ]
        return input_ids

    def get_logits(self, model, base="", attack=""):
        # Get the logits tensor for the conversation prompt

        input_ids = self.get_input_ids(base=base, attack=attack)
        logits = model(input_ids.unsqueeze(0))[0]

        return logits
