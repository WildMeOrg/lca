
class ClassifierBase(object):
    def __init__(self, name):
        self.name = name
        

    def __call__(self, edge):
        raise NotImplemented()