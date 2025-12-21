from models.model import Model
from typing import List

class ModelStack:

    def __init__(self, models:List[Model]):
        self.types = [model.model_type for model in models]

        self.models = models

    def run(self, frame):
        """
        Model run meant to record the output of each model and append it to a list that can be analyzed or
        displayed with a later function.
        """
        outputs = []
        output = None
        for model in self.models:
            model_inputs = model.prepare_inputs(output, frame)
            model_outputs = model.predict(model_inputs)
            output = model.prepare_outputs(model_outputs)
            outputs.append(output)
            if len(output)==0:
                break

        return outputs

    def run_final_output(self, frame):
        """
        Model run for just getting the final output of a model stack. Meant to a more efficient way of processing
        the model outputs if needed.
        """
        output = None
        for model in self.models:
            model_inputs, frame = model.prepare_inputs(output, frame)
            model_outputs = model.predict(model_inputs)
            output = model.prepare_outputs(model_outputs)
            output.append(outputs)
            if len(output) == 0:
                break

        return output

