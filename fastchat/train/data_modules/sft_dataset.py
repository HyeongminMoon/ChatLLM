from typing import Dict, Optional, List, Union
from datasets import Dataset, load_dataset

from fastchat.model.model_adapter import get_conversation_template

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
    
    