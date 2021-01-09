from typing import Dict
from abc import ABC


# @TODO: do we really need it?
# @TODO: or ILogger is good enough?
class ITrial(ABC):
    """
    An abstraction that syncs experiment run with
    different hyperparameter-search systems.
    """

    pass


class Trial(ITrial):
    pass


def get_trial_by_params(trial_params: Dict):
    return Trial()
