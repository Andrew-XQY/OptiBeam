from .utils import *


class Converter:
    """
    Class used to convert generated beam pattern (image) into different formats including Python objects and files.
    where divices like DMD, SLM can load the beam pattern.
    """
    def __init__(self, name, input_type, output_type, function):
        self.name = name
        self.input_type = input_type
        self.output_type = output_type
        self.function = function

    def convert(self, value):
        return self.function(value)


class Simulation:
    def __init__(self, beam_pattern, converter):
        self.beam_pattern = beam_pattern