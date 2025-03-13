from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch

from clfextract.utils import type_check


class ParquetManager:
    """
    Class to handle the Parquet datasets
    """

    def __init__(self):
        return

    # Case-specific method to create Parquet files, not ideal
    @type_check
    def create_parquet(
        self,
        tensor: torch.Tensor,
        labels: torch.Tensor,
        emb_type: str,
        output_file: str,
        prompts: Optional[List[str]] = None,
    ):
        """
        Creates a Parquet file from the given tensor and labels.
        Args:
            tensor (torch.Tensor): The input tensor containing the data.
            labels (torch.Tensor): The tensor containing the labels.
            emb_type (str): The type of embeddings. Must be one of ["embeddings", "keys", "values", "attentions"].
            output_file (str): The path to the output Parquet file.
            prompts (Optional[List[str]], optional): A list of prompts corresponding to each sample. Defaults to None.
        Returns:
            None
        """
        assert emb_type in ["embeddings", "keys", "values", "attentions"]
        data = []

        y_true = labels // 2
        y_pred = labels % 2

        if emb_type in ["keys", "values", "attentions"]:
            num_samples, num_layers, num_heads, num_positions, hidden_size = (
                tensor.shape
            )
            for sample in range(num_samples):
                for layer in range(num_layers):
                    for head in range(num_heads):
                        data.append(
                            {
                                "prompt": (
                                    prompts[sample] if prompts is not None else None
                                ),
                                "layer": layer,
                                "head": head,
                                "x": tensor[sample, layer, head, :, :]
                                .numpy()
                                .tobytes(),
                                "num_positions": num_positions,
                                "y_true": int(y_true[sample]),
                                "y_pred": int(y_pred[sample]),
                            }
                        )
        else:  # embeddings
            num_samples, num_layers, num_positions, hidden_size = tensor.shape
            for sample in range(num_samples):
                for layer in range(num_layers):
                    data.append(
                        {
                            "prompt": prompts[sample] if prompts is not None else None,
                            "layer": layer,
                            "x": tensor[sample, layer, :, :].numpy().tobytes(),
                            "num_positions": num_positions,  # Prob need to create dynamically because num_positions is not always the same
                            "y_true": int(y_true[sample]),
                            "y_pred": int(y_pred[sample]),
                        }
                    )

        df = pl.DataFrame(data)
        df.write_parquet(output_file)

        return None

    def load_dataset(
        self,
        parquet_file: str,
        filters: Dict[str, Any] = None,
        columns: List[str] = None,
    ) -> pl.DataFrame:
        """
        Load a dataset from a Parquet file with optional filtering and column selection.
        Args:
            parquet_file (str): The path to the Parquet file to load.
            filters (Dict[str, Any], optional): A dictionary of filters to apply to the dataset. Defaults to None.
            columns (List[str], optional): A list of column names to select from the dataset. Defaults to None.
        Returns:
            pl.DataFrame: A Polars DataFrame containing the loaded dataset.
        """

        lazy_data = pl.scan_parquet(parquet_file)

        if filters:
            lazy_data = self.apply_filters(lazy_data, filters)

        if columns:
            lazy_data = lazy_data.select(columns)

        return lazy_data.collect()

    @staticmethod
    def apply_filters(lazy_data: pl.LazyFrame, filters: Dict[str, Any]) -> pl.LazyFrame:
        """
        Args:
            lazy_data (pl.LazyFrame): The LazyFrame with the dataset.
            filters (Dict[str, Any], optional): A dictionary of filters to apply to the dataset.
        Returns:
            pl.LazyFrame: A Polars LazyFrame with the filtered dataset.

        """
        for column, condition in filters.items():
            if isinstance(condition, dict):
                # Complex filtering
                for op, value in condition.items():
                    match op:
                        case "contains":
                            lazy_data = lazy_data.filter(
                                pl.col(column).str.contains(value)
                            )
                        case "between":
                            assert (
                                isinstance(value, list) and len(value) == 2
                            ), f"Invalid value for between condition: {value}"
                            lazy_data = lazy_data.filter(
                                (pl.col(column) >= value[0])
                                & (pl.col(column) <= value[1])
                            )
                        case "equals":
                            lazy_data = lazy_data.filter(pl.col(column) == value)
                        case "not_equals":
                            lazy_data = lazy_data.filter(pl.col(column) != value)
                        case "greater_than":
                            lazy_data = lazy_data.filter(pl.col(column) > value)
                        case "less_than":
                            lazy_data = lazy_data.filter(pl.col(column) < value)
                        case "in":
                            lazy_data = lazy_data.filter(pl.col(column).is_in(value))
                        case _:
                            raise ValueError(f"Invalid operation {op}")
            elif isinstance(condition, list):
                # List of values (OR condition)
                lazy_data = lazy_data.filter(pl.col(column).is_in(condition))
            else:
                # Simple equality check
                lazy_data = lazy_data.filter(pl.col(column) == condition)

        # Special case for y_true == y_pred
        if "y_true_equals_y_pred" in filters and filters["y_true_equals_y_pred"]:
            lazy_data = lazy_data.filter(pl.col("y_true") == pl.col("y_pred"))

        return lazy_data

    @staticmethod
    def merge_parquet_files(input_files: List[str], output_file: str):
        assert len(input_files) > 0, "No input files provided"
        assert all(
            [file.endswith(".parquet") for file in input_files]
        ), "All input files must be Parquet files"
        assert output_file.endswith(".parquet"), "Output file must be a Parquet file"
        # Create a list of lazy DataFrames
        lazy_dfs = [pl.scan_parquet(file) for file in input_files]
        # Concatenate all lazy DataFrames
        merged_df = pl.concat(lazy_dfs)
        # Write the merged data to a new Parquet file
        merged_df.sink_parquet(output_file)

    @staticmethod
    def lazy_edit_column(
        input_file: str,
        output_file: str,
        edit_functions: Dict[str, Callable[[pl.Expr], pl.Expr]],
    ):
        """
        Applied specific functions to the dataset in a file
        Args:
            input_file (str): Path to the input file
            output_file (str): Path to the output file
            edit_functions (Dict[str, Callable[[pl.Expr], pl.Expr]]): A dictionary of column names and edit functions

        """
        assert input_file.endswith(".parquet"), "Input file must be a Parquet file"
        assert output_file.endswith(".parquet"), "Output file must be a Parquet file"

        # Read the input Parquet file
        lazy_df = pl.scan_parquet(input_file)

        # Apply the edit functions to the specified columns
        edited_columns = [
            edit_function(pl.col(column)).alias(column)
            for column, edit_function in edit_functions.items()
        ]

        lazy_df = lazy_df.with_columns(edited_columns)

        # Write the edited data to a new Parquet file
        lazy_df.sink_parquet(output_file)
        return

    @staticmethod
    def concatenate_layers(
        df: pl.DataFrame, layers: Optional[List[int]] = None, mode: str = "sequence"
    ) -> pl.DataFrame:
        """
        Concatenate 'x' tensors across specified layers for each unique prompt.

        Args:
        df (pl.DataFrame): The input DataFrame containing 'x', 'prompt', 'layer', and other columns.
        layers (List[int]): The layers to include in the concatenation.

        Returns:
        pl.DataFrame: A DataFrame with concatenated 'x' values and updated 'num_positions' and 'hidden_size'.
        """
        # Filter the DataFrame to include only the specified layers
        if layers is not None:
            df = df.filter(pl.col("layer").is_in(layers))

        # Sort the DataFrame by prompt and layer to ensure correct ordering
        df = df.sort(["prompt", "layer"])
        # IMPORTANT Remove duplicate prompts, otherwise concatenation will fail later
        df = df.unique(subset=["prompt", "layer"])

        df = df.group_by("prompt").agg(
            ["x", pl.first("num_positions"), pl.first("y_true"), pl.first("y_pred")]
        )

        def concatenate_row(row: pl.Series) -> pl.Series:
            x_list = row["x"]
            num_positions = row["num_positions"]

            # Concatenate the binary data
            if mode == "sequence":
                concatenated_x = np.concatenate(
                    [
                        np.expand_dims(
                            np.frombuffer(x, dtype=np.float32).reshape(
                                num_positions, -1
                            ),
                            axis=1,
                        )
                        for x in x_list
                    ],
                    axis=1,
                )
            else:
                concatenated_x = np.concatenate(
                    [
                        np.frombuffer(x, dtype=np.float32).reshape(num_positions, -1)
                        for x in x_list
                    ],
                    axis=-1,
                )

            return concatenated_x.tobytes()

        # Apply the concatenation function to each row
        result = df.select(
            [
                pl.col("prompt"),
                pl.struct(["x", "num_positions"])
                .map_elements(concatenate_row, return_dtype=pl.datatypes.Binary)
                .alias("x"),
                pl.col("num_positions"),
                pl.col("y_true"),
                pl.col("y_pred"),
            ]
        )

        return result

    @staticmethod
    def concatenate_layers_torch(
        df: pl.DataFrame,
        layers: Optional[List[int]] = None,
        aggregation: str = "last",
        mode: str = "fixed",
        device: str = "cpu",
    ) -> Tuple[torch.Tensor]:
        """
        Concatenate 'x' tensors across specified layers for each unique prompt. Return as PyTorch tensors.

        Args:
        df (pl.DataFrame): The input DataFrame containing 'x', 'prompt', 'layer', and other columns.
        layers (List[int]): The layers to include in the concatenation.
        aggregation (str): The aggregation method to use for combining the layers.
        mode (str): The mode to use for concatenation ('sequence' or 'fixed').
        device (str): The device to use for the output tensors


        Returns:
        Tuple[torch.Tensor]: A tuple containing the concatenated 'x' values, 'y_true', and 'y_pred'.
        """
        # Filter the DataFrame to include only the specified layers
        if layers is not None:
            df = df.filter(pl.col("layer").is_in(layers))
            num_layers = len(layers)
        else:
            # Count the number of unique layers
            num_layers = df.n_unique(subset=["layer"])

        # Sort the DataFrame by prompt and layer to ensure correct ordering
        df = df.sort(["prompt", "layer"])
        # IMPORTANT Remove duplicate prompts, otherwise concatenation will fail later
        df = df.unique(subset=["prompt", "layer"])

        df = df.group_by("prompt").agg(
            ["x", pl.first("num_positions"), pl.first("y_true"), pl.first("y_pred")]
        )

        def concatenate_row(row: pl.Series) -> pl.Series:
            x_list = row["x"]
            num_positions = row["num_positions"]

            # Concatenate the binary data
            match mode:
                case "sequence":
                    concatenated_x = np.concatenate(
                        [
                            np.expand_dims(
                                np.frombuffer(x, dtype=np.float32).reshape(
                                    num_positions, -1
                                ),
                                axis=1,
                            )
                            for x in x_list
                        ],
                        axis=1,
                    )
                case "fixed":
                    concatenated_x = np.concatenate(
                        [
                            np.frombuffer(x, dtype=np.float32).reshape(
                                num_positions, -1
                            )
                            for x in x_list
                        ],
                        axis=-1,
                    )

                case _:
                    raise ValueError(f"Invalid mode {mode}")

            return concatenated_x.tobytes()

        # Apply the concatenation function to each row
        df = df.select(
            [
                pl.col("prompt"),
                pl.struct(["x", "num_positions"])
                .map_elements(concatenate_row, return_dtype=pl.datatypes.Binary)
                .alias("x"),
                pl.col("num_positions"),
                pl.col("y_true"),
                pl.col("y_pred"),
            ]
        )

        match aggregation:
            case "mean":  # Average over all positions
                x = np.stack(
                    [
                        np.frombuffer(row["x"], dtype=np.float32)
                        .reshape(row["num_positions"], num_layers, -1)
                        .mean(axis=0)
                        for row in df.iter_rows(named=True)
                    ]
                )

            case "last":  # Take the last position
                x = np.stack(
                    [
                        np.frombuffer(row["x"], dtype=np.float32).reshape(
                            row["num_positions"], num_layers, -1
                        )[-1, :, :]
                        for row in df.iter_rows(named=True)
                    ]
                )

            case "padding":  # Zero-pad to the maximum number of positions
                max_num_positions = df["num_positions"].max()
                x = []
                for tensor, row in zip(df["x"], df.iter_rows(named=True)):
                    array = np.frombuffer(tensor, dtype=np.float32).reshape(
                        row["num_positions"], num_layers, -1
                    )
                    padded_array = np.zeros(
                        (max_num_positions, num_layers, -1), dtype=np.float32
                    )
                    padded_array[row["num_positions"], :, :] = array
                    x.append(padded_array)
                x = np.stack(x)

            case None:  # No aggregation
                x = [
                    np.frombuffer(row["x"], dtype=np.float32).reshape(
                        row["num_positions"], num_layers, -1
                    )
                    for row in df.iter_rows(named=True)
                ]
                y_true = torch.tensor(df["y_true"].to_numpy()).to(device)
                y_pred = torch.tensor(df["y_pred"].to_numpy()).to(device)
                return (
                    [torch.from_numpy(array).to(device) for array in x],
                    y_true,
                    y_pred,
                )

            case _:
                raise ValueError(f"Invalid aggregation method {aggregation}")

        x = torch.from_numpy(x).to(device)
        y_true = torch.tensor(df["y_true"].to_numpy()).to(device)
        y_pred = torch.tensor(df["y_pred"].to_numpy()).to(device)
        return x, y_true, y_pred

    @staticmethod
    def load_torch_from_df(
        df: pl.DataFrame,
        aggregation: Optional[str] = None,
        device: str = "cpu",
        columns: List[str] = ["x", "y_true", "y_pred"],
    ) -> Tuple[torch.Tensor]:
        """
        Loads data from a Polars DataFrame into PyTorch tensors.
        This function converts data stored in a Polars DataFrame into PyTorch tensors,
        with optional aggregation for sequential data stored in the 'x' column.
        Args:
            df (pl.DataFrame): Input Polars DataFrame containing the data.
            aggregation (Optional[str], optional): Aggregation method for sequential data.
                Supported values:
                - 'mean': Average over all positions
                - 'last': Take the last position
                - 'padding': Zero-pad to maximum number of positions
                - None: No aggregation, returns list of tensors
                Defaults to None.
            device (str, optional): PyTorch device to store tensors on. Defaults to "cpu".
            columns (List[str], optional): List of column names to process.
                Defaults to ["x", "y_true", "y_pred"].
        Returns:
            Tuple[torch.Tensor]: Tuple of PyTorch tensors containing the processed data.
                The tensors correspond to the columns in the input order.
                For non-aggregated data (aggregation=None), returns a tuple containing:
                - List of torch.Tensor for 'x' column (if present)
                - numpy arrays for other specified columns
        Raises:
            ValueError: If an invalid aggregation method is specified.
        """
        column_values = []
        if "x" in columns:
            match aggregation:
                case "mean":  # Average over all positions
                    x = np.stack(
                        [
                            np.frombuffer(row["x"], dtype=np.float32)
                            .reshape(row["num_positions"], -1)
                            .mean(axis=0)
                            for row in df.iter_rows(named=True)
                        ]
                    )

                case "last":  # Take the last position
                    x = np.stack(
                        [
                            np.frombuffer(row["x"], dtype=np.float32).reshape(
                                row["num_positions"], -1
                            )[-1, :]
                            for row in df.iter_rows(named=True)
                        ]
                    )

                case "padding":  # Zero-pad to the maximum number of positions
                    max_num_positions = df["num_positions"].max()
                    x = []
                    for tensor, row in zip(df["x"], df.iter_rows(named=True)):
                        array = np.frombuffer(tensor, dtype=np.float32).reshape(
                            row["num_positions"], -1
                        )
                        padded_array = np.zeros(
                            (max_num_positions, -1), dtype=np.float32
                        )
                        padded_array[: row["num_positions"], :] = array
                        x.append(padded_array)
                    x = np.stack(x)

                case None:  # No aggregation
                    x = [
                        np.frombuffer(row["x"], dtype=np.float32).reshape(
                            row["num_positions"], -1
                        )
                        for row in df.iter_rows(named=True)
                    ]
                    column_values.append(
                        [torch.from_numpy(array).to(device) for array in x]
                    )

                    for column in columns[1:]:
                        column_values.append(df[column].to_numpy())
                    return tuple(column_values)
                case _:
                    raise ValueError(f"Invalid aggregation method {aggregation}")

            x = torch.from_numpy(x).to(device)
            column_values.append(x)

        for column in columns:
            if column == "x":
                continue
            column_values.append(torch.tensor(df[column].to_numpy()).to(device))
        return tuple(column_values)
