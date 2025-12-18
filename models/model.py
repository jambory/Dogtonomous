from abc import ABC, abstractmethod
from typing import Dict

class Model(ABC):
    context:Dict
    model_type:str

    @abstractmethod
    def prepare_inputs(self, inputs, frame):
        """
        Abstract method for preparing inputs for model.
        """
        raise NotImplemented

    @abstractmethod
    def predict(self, inputs):
        """
        Abstract method for predict on inputs of model.
        """
        raise NotImplemented

    @abstractmethod
    def prepare_outputs(self, output):
        """
        Abstract method for formatting/preparing outputs of model.
        """
        raise NotImplemented
