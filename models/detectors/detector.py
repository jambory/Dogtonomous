from models.model import Model

class Detector(Model):
    def __init__(self):
        self.model_type = 'detector'
        self.context = {}
        self.conf_thresh:float

    def prepare_inputs(self, inputs, frame):
        """
        For almost any detector, the only input will be the image data itself. Thus a very
        simple prepre inputs method.
        """
        return frame

    def prepare_outputs(self, output):
        """
        Abstract method for formatting/preparing outputs of model.
        """
        raise NotImplemented

    def predict(self, inputs):
        """
        Abstract method for predict on inputs of model.
        """
        raise NotImplemented


if __name__ =='__main__':
    classifier = Classifier()