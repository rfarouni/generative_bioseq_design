from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from pandas import DataFrame
import os
import numpy as np

class SeqDataset(Dataset):
    def __init__(self, df):
        self.sequences = df['seq'].tolist()
        self.labels = df['label'].tolist()
        self.amino_acid_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
         # Create a dictionary to map amino acids to their one-hot encoding
        self.encoding_dict = {aa: np.eye(len(self.amino_acid_list))[i].astype(int).tolist() for i, aa in enumerate(self.amino_acid_list)}
        
    def __len__(self):
        return len(self.sequences)
    
    def encode_sequence(self, sequence):
        encoded_sequence = [self.encoding_dict[base] for base in sequence]
        encoded_sequence = [item for sublist in encoded_sequence for item in sublist]
        return encoded_sequence
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        label=torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        encoded_sequence = self.encode_sequence(sequence)
        encoded_sequence = torch.tensor(encoded_sequence, dtype=torch.float32)
        #encoded_sequence =encoded_sequence.flatten()
        return encoded_sequence, label
     
class AAVSeqDatasetModule(LightningDataModule):
    """`LightningDataModule` for the AAV dataset.


    A `LightningDataModule` implements 7 key methods:
    
    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """
    
    def __init__(
        self,
        data_dir: str = "data/aav9_6mer",
        dataset_name: str = "dataset.tsv",
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        batch_size: int = 512,
        num_workers: int = 10,
        pin_memory: bool = True,
    ) -> None:
        """Initialize a `DataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split.
        :param batch_size: The batch size. Defaults to `512`.
        :param num_workers: The number of workers. Defaults to `10`.
        :param pin_memory: Whether to pin memory. Defaults to `True`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.dataframe = None
        self.dataset = None


        self.batch_size_per_device = batch_size
    
    def load_data(self, df_filepath : str) -> DataFrame:
        dataframe = pd.read_csv(os.path.join(df_filepath), sep='\t')
        dataframe['label'] = dataframe['label'].astype(int)
        return dataframe
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            df_filepath = os.path.join(self.hparams.data_dir, self.hparams.dataset_name)
            self.dataframe = self.load_data(df_filepath)
            self.dataset = SeqDataset(self.dataframe)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=self.dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

if __name__ == "__main__":
    _ = AAVSeqDatasetModule()
