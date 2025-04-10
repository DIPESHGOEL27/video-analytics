import yaml
import os

def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Parsed configuration as a dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML config: {e}")

    return config
