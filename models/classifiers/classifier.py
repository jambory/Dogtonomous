from models.model import Model

class Classifier(Model):

    def __init__(self):
        self.model_type = 'classifier'
        self.context = {}
        self.conf_thresh: float

    def prepare_inputs(self, inputs, frame):
        """
        Abstract method for preparing inputs for model.
        """
        raise NotImplemented

    def predict(self, inputs):
        """
        Abstract method for predict on inputs of model.
        """
        raise NotImplemented

    def prepare_outputs(self, output):
        """
        Abstract method for formatting/preparing outputs of model.
        """
        raise NotImplemented