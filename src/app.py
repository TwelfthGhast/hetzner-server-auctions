from hetzner import HetznerAuction
import pandas as pd

pd.set_option('display.max_colwidth', 100)

obj = HetznerAuction()
print(obj.filter(ram_max=128, ram_min=16))