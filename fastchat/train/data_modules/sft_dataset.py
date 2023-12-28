from typing import Dict, Optional, List, Union
from datasets import Dataset, load_dataset

from fastchat.model.model_adapter import get_conversation_template

def load_sft_dataset(dataset_path, split='train'):
    split='train'
    dataset = load_dataset("json", data_files=dataset_path, split=split)

    if "id" not in dataset.features:
        dataset = dataset.map(lambda x: {"id": "null"})

    if hasattr(dataset.features["id"], "dtype") and dataset.features["id"].dtype == "int64":
        dataset = dataset.map(lambda x: {"id": "id" + str(x["id"])})
    
    return dataset

class WizardVicunaOrcaDataset:
    """
    WizardLM + VicunaLM + Orca-Chat.
    """
    def __init__(
        self, 
        dataset_path: Union[List[str], str] = "/data/llm_datasets/Ultrafeedback_binarized.ko.hankang/",
        data_format='chat-orca',
        num_train=None,
        num_eval=None,
    ):
        self.dataset_path = dataset_path
        self.data_format = data_format
        self.num_train = num_train
        self.num_eval = num_eval
    
    