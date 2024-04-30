import os, yaml

def Setting():
    # log config file
    env_var = os.environ.get("WHICH_CONFIG")
    
    config_files = {
        "local.yaml":"local.yaml",
        # Add more mappings as needed
    }
    
    if env_var in config_files:
        yaml_config_file = os.path.join("env", config_files[env_var])
        with open(yaml_config_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    else:
        config = "env.yaml"
        
    return config
