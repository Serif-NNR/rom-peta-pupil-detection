
import os
import importlib
import torch

def makedir(dir):
    """
    Creates directory if it does not exist.

    :param dir: directory path
    :type dir: str
    """

    if dir and not os.path.exists(dir):
        os.makedirs(dir)


def get_class(module_name, class_name):
    """
    See https://stackoverflow.com/questions/1176136/convert-string-to-python-class-object?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa.

    :param module_name: module holding class
    :type module_name: str
    :param class_name: class name
    :type class_name: str
    :return: class or False
    """
    # load the module, will raise ImportError if module cannot be loaded
    try:
        m = importlib.import_module(module_name)
    except ImportError as e:
        return False
    # get the class, will raise AttributeError if class cannot be found
    try:
        c = getattr(m, class_name)
    except AttributeError as e:
        return False
    return c


class State(object):
        """
        State of a model, including optional epoch and optimizer.
        """

        # 1. A state consists of a model, optimizer, scheduler and an epoch.
        # This allows to resume training given the used optimizer, its internal state, and the scheduler.
        def __init__(self, model, optimizer=None, scheduler=None, epoch=None):
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.epoch = epoch

        # 2. Saves the current state:
        # - The model class name is saved
        # - Parametrs, optimizer, scheduler and epoch are added.
        # - All attributes of the model class that do not start with _ or __ are saved (these are assumed to be the model's hyper-parameters).
        # - All keyword arguments of the model are saved as well.
        def save(self, filepath):
            model = self.model
            if not isinstance(model, dict):
                model = model.state_dict()
            model_class = self.model.__class__.__name__

            optimizer = self.optimizer
            if not isinstance(optimizer, dict) and optimizer is not None:
                optimizer = optimizer.state_dict()

            scheduler = self.scheduler
            if not isinstance(scheduler, dict) and scheduler is not None:
                scheduler = scheduler.state_dict()

            epoch = self.epoch
            #assert get_class('models', model_class) is not False
            arguments = dict((key, getattr(self.model, key)) for key in dir(self.model)
                             if not callable(getattr(self.model, key)) and not key.startswith('_') and not key == 'kwargs' and not key == 'T_destination')
            kwargs = getattr(self.model, 'kwargs', None)
            makedir(os.path.dirname(filepath))

            data = {'model': model, 'model_class': model_class,
                    'optimizer': optimizer, 'scheduler': scheduler,
                    'epoch': epoch, 'arguments': arguments, 'kwargs': kwargs}

            torch.save(data, filepath)

        # 3. Loading a model:
        # - The architecture is instantiated using the saved class name and the saved arguments and keyword arguments.
        # - A new state is created using the model (after loading paramteres), optimizer, scheduler and epoch.
        # Note that optimizer, scheduler and epoch may be None.
        @staticmethod
        def load(filepath, class_structure):
            assert os.path.exists(filepath), 'file %s not found' % str(filepath)

            # https://discuss.pytorch.org/t/gpu-memory-usage-increases-by-90-after-torch-load/9213/3
            checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

            model_class = class_structure #get_class('models', checkpoint['model_class'])
            if 'kwargs' in checkpoint and False:
                arguments = {**checkpoint['arguments'], **checkpoint['kwargs']}
            else:
                arguments = {**checkpoint['arguments']}
            model = model_class(**arguments)
            model.load_state_dict(checkpoint['model'], strict=False)

            state = State(model, checkpoint['optimizer'], checkpoint['scheduler'], checkpoint['epoch'])

            del checkpoint
            torch.cuda.empty_cache()

            return state

        @staticmethod
        def checkpoint(filepath, model, optimizer=None, scheduler=None, epoch=None):
            state = State(model, optimizer, scheduler, epoch)
            state.save(filepath)