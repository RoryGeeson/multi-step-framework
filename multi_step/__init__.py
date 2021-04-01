import os
import json

backend_name = 'tensorflow'
default_dir = os.path.join(os.path.expanduser('~'), '.dgl')
if not os.path.exists(default_dir):
    os.makedirs(default_dir)
config_path = os.path.join(default_dir, 'config.json')
with open(config_path, "w") as config_file:
    json.dump({'backend': backend_name.lower()}, config_file)
print('Setting the default backend to "{}". You can change it in the '
      '~/.dgl/config.json file or export the DGLBACKEND environment variable.  '
      'Valid options are: pytorch, mxnet, tensorflow (all lowercase)'.format(
          backend_name))