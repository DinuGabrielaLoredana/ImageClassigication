class ModelInterface(object):
    def __init__(self):
        pass

    def read_dataset(self):
        raise Exception("NotImplementedException")

    def augment_data(self):
        raise Exception("NotImplementedException")

    def compile_model(self):
        raise Exception("NotImplementedException")

    def train(self):
        raise Exception("NotImplementedException")

    def save_model(self):
        raise Exception("NotImplementedException")
