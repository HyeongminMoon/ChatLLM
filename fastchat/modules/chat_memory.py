from fastchat.model.model_adapter import get_conversation_template


class BaseMemoryBlock:
    def __init__(self):
        self.raw = ""

    def load(self):
        return self.conv

    def save(self, text):
        self.raw = text


class SummaryMemoryBlock(BaseMemoryBlock):
    def __init__(self):
        self.data = ""

    def load(self):
        return self.conv

    def save(self, text):
        self.raw = text


class MemoryGroup:
    def __init__(self, model_name):
        self.chat_memory = BaseMemoryBlock(model_name)
        self.websearch_memory = BaseMemoryBlock("")
        self.translation_memory = BaseMemoryBlock("")
