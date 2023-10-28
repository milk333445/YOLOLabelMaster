import yaml
from easydict import EasyDict



class ConfigManager:
       def __init__(self, filepath='./autolabel_settings.yaml'):
              with open(filepath, 'r') as yaml_file:
                     self.data = yaml.safe_load(yaml_file)
              if 'alphabet_numbers' in self.data:
                  # 轉換成字串
                     self.data['alphabet_numbers'] = [str(x) for x in self.data['alphabet_numbers']]

       def get_config(self):
            conf = {} # EasyDict()沒辦法處理 key_actions
            conf['obj'] = self.data['alphabet_numbers']
            conf['drawing'] = self.data['settings']['drawing']

            conf['clr'] = [tuple(color) for color in self.data['settings']['clr']]
            conf['key_actions'] = {int(k): v for k, v in self.data['key_actions'].items()} # Convert string keys to integers
            return conf






