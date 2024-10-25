import toml
 
with open('ip_file.toml', 'r') as f:
    config = toml.load(f)
 
# Access values from the config
print(config['victim'])