from hetzner import HetznerAuction
import pandas as pd

pd.set_option('display.max_colwidth', 100)

obj = HetznerAuction()
print(obj.ram(max=128, min=16).ssd(min=100).ecc(True).sort(how="price_asc"))