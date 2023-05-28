import yaml
from box import Box


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = Box(yaml.safe_load(f))
    return config
