import requests
import json
import numpy as np
import pandas as pd
import math

# representation of standard 4 byte int
MAX_INT = int(math.pow(2, 31) - 1)

class HetznerAuction():
    def __init__(self):
        r = requests.get("https://www.hetzner.com/a_hz_serverboerse/live_data.json")
        data = r.json()
        # solve linear matrix
        coefficient_matrix = []
        ordinate_vars = []

        df = pd.DataFrame(columns=["value", "price", "expected_price", "ram", "is_ecc", "ssd", "hdd", "cpu", "cpu_score"])

        for server in data["server"]:
            cpu_bench = server["cpu_benchmark"]
            price = server["price"]
            ecc = server["is_ecc"]
            ram = int(server["ram"])
            hdd_size = 0
            ssd_size = 0
            for item in server["description"]:
                if item.split()[1] == "HDD":
                    qty = int(item.split()[0][:-1])
                    modifier = 1 
                    if item.split()[4] == "TB":
                        modifier = 1024
                    space = float(item.split()[3].replace(",", "."))
                    hdd_size += qty*modifier * space
                elif item.split()[1] == "SSD":
                    qty = int(item.split()[0][:-1])
                    modifier = 1 
                    try:
                        space = float(item.split()[3].replace(",", "."))
                        if item.split()[4] == "TB":
                            modifier = 1024
                    except ValueError:
                        space = float(item.split()[4].replace(",", "."))
                        if item.split()[5] == "TB":
                            modifier = 1024
                    ssd_size += qty*modifier * space
            cpu = server["cpu"]
            description = ", ".join(server["description"])
            df = df.append({
                "price": price,
                "is_ecc": 1 if ecc else 0,
                "ram": ram,
                "ssd": ssd_size,
                "hdd": hdd_size,
                "cpu": cpu,
                "cpu_score": cpu_bench,
                "description": description
            }, ignore_index=True)

        # clean up data
        for index, row in df.iterrows():
            # rarely cpu_score is 0
            if row["cpu_score"] == 0:
                cpu = row["cpu"]
                cpu_score = df.loc[(df["cpu_score"] > 0) & (df["cpu"] == cpu)]["cpu_score"].min()
                df.loc[index, "cpu_score"] = cpu_score
            coefficient_matrix.append([row["cpu_score"], row["is_ecc"], row["ram"], row["hdd"], row["ssd"]])
            ordinate_vars.append(row["price"])

        a = np.array(coefficient_matrix, dtype='float')
        b = np.array(ordinate_vars, dtype='float')
        x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)

        cpu_bench_modifier, ecc_modifier, ram_modifier, hdd_modifier, ssd_modifier = x
        
        for index, row in df.iterrows():
            ev = row["cpu_score"]*cpu_bench_modifier + row["ram"]*ram_modifier + row["hdd"]*hdd_modifier + row["ssd"]*ssd_modifier + row["is_ecc"]*ecc_modifier
            df.loc[index, "expected_price"] = ev
            df.loc[index, "value"] = ev / float(row["price"])

        df.sort_values("value", ascending=False, inplace=True)
        df["price"] = df.price.astype(float)

        self._data = AuctionData(df.reset_index(drop=True))
    
    @property
    def data(self):
        return self._data
    
    def __str__(self):
        return str(self._data)

class AuctionData():
    def __init__(self, df):
        self._data = df

    @property
    def data(self):
        return self._data.copy()
    
    def filter(self,
        ram_min: int = 0, ram_max: int = MAX_INT,
        is_ecc: bool = None,
        storage_min: int = 0, storage_max: int = MAX_INT,
        ssd_min: int = 0, ssd_max: int = MAX_INT,
        price_min: float = 0.0, price_max: float = float(MAX_INT)
    ):
        df = self.data
        df = df[df.ram >= ram_min][df.ram <= ram_max]
        if is_ecc is not None:
            if is_ecc:
                df = df[df.is_ecc == 1]
            else:
                df = df[df.is_ecc == 0]
        df = df[(df.hdd + df.ssd) >= storage_min][(df.hdd + df.ssd) <= storage_max]
        df = df[df.ssd >= ssd_min][df.ssd <= ssd_max]
        df = df[df.price >= price_min][df.price <= price_max]
        return AuctionData(df.reset_index(drop=True))
    
    def __str__(self):
        return str(self._data.copy())
