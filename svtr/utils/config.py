import yaml

def load_yaml(file_path):
    try:
        with open(file_path, 'r') as f:
            print(f"Reading option file: {file_path}")
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    except PermissionError:
        print(f"Error: Permission denied - {file_path}")
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file - {e}")
    return None

class Config:
    def __init__(self, config_path='config/config.yml'):
        self.config = load_yaml(config_path)
        self.global_config = self.config['global']
        self.dataset_config = self.config['dataset']