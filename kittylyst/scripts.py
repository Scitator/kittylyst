import fire
import yaml

from kittylyst.experiment import ConfigExperiment
from kittylyst.runner import SupervisedRunner


def run(config):
    with open(config) as stream:
        config_dict = yaml.load(stream, yaml.Loader)

    experiment = ConfigExperiment(config_dict)
    SupervisedRunner().run(experiment)


def main():
    fire.Fire({"run": run})


if __name__ == "__main__":
    main()
