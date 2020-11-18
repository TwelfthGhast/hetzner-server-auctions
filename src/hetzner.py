import requests
import json
import numpy as np
import pandas as pd
import math
from cachetools import cached, TTLCache

# representation of standard 4 byte int
MAX_INT = int(math.pow(2, 31) - 1)

@cached(cache=TTLCache(maxsize=1, ttl=60*5))
def get_server_data() -> pd.DataFrame:
    r = requests.get("https://www.hetzner.com/a_hz_serverboerse/live_data.json")
    data = r.json()

    df = pd.DataFrame(
        columns=[
            "value",
            "price",
            "expected_price",
            "ram",
            "is_ecc",
            "ssd",
            "hdd",
            "cpu",
            "cpu_score",
        ]
    )

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
                data = item.split()
                modifier = 1
                space = 0
                for i, val in enumerate(data):
                    if val in ["GB", "TB"]:
                        if val == "TB":
                            modifier = 1024
                        space = float(data[i - 1].replace(",", "."))
                hdd_size += qty * modifier * space
            elif item.split()[1] == "SSD":
                data = item.split()
                qty = int(item.split()[0][:-1])
                modifier = 1
                space = 0
                for i, val in enumerate(data):
                    if val in ["TB", "GB"]:
                        if val == "TB":
                            modifier = 1024
                        space = float(data[i - 1].replace(",", "."))
                ssd_size += qty * modifier * space
        cpu = server["cpu"]
        description = ", ".join(server["description"])
        df = df.append(
            {
                "price": price,
                "is_ecc": 1 if ecc else 0,
                "ram": ram,
                "ssd": ssd_size,
                "hdd": hdd_size,
                "cpu": cpu,
                "cpu_score": cpu_bench,
                "description": description,
            },
            ignore_index=True,
        )
    return df


class HetznerAuction:
    def __init__(self):
        df = get_server_data()
        # solve linear matrix
        coefficient_matrix = []
        ordinate_vars = []
        # clean up data
        for index, row in df.iterrows():
            # rarely cpu_score is 0
            if row["cpu_score"] == 0:
                cpu = row["cpu"]
                cpu_score = df.loc[(df["cpu_score"] > 0) & (df["cpu"] == cpu)][
                    "cpu_score"
                ].min()
                df.loc[index, "cpu_score"] = cpu_score
            coefficient_matrix.append(
                [row["cpu_score"], row["is_ecc"], row["ram"], row["hdd"], row["ssd"]]
            )
            ordinate_vars.append(row["price"])

        a = np.array(coefficient_matrix, dtype="float")
        b = np.array(ordinate_vars, dtype="float")
        x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)

        cpu_bench_modifier, ecc_modifier, ram_modifier, hdd_modifier, ssd_modifier = x

        for index, row in df.iterrows():
            ev = (
                row["cpu_score"] * cpu_bench_modifier
                + row["ram"] * ram_modifier
                + row["hdd"] * hdd_modifier
                + row["ssd"] * ssd_modifier
                + row["is_ecc"] * ecc_modifier
            )
            df.loc[index, "expected_price"] = ev
            df.loc[index, "value"] = ev / float(row["price"])

        df = df.sort_values("value", ascending=False)
        df["price"] = df.price.astype(float)

        self._data = df.reset_index(drop=True)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, df: pd.DataFrame):
        self._data = df

    def ram(self, **kwargs):
        df = self.data
        if "max" in kwargs:
            df = df[df.ram <= kwargs["max"]]
        if "min" in kwargs:
            df = df[df.ram >= kwargs["min"]]
        self.data = df.reset_index(drop=True)
        return self

    def ssd(self, **kwargs):
        df = self.data
        if "max" in kwargs:
            df = df[df.ssd <= kwargs["max"]]
        if "min" in kwargs:
            df = df[df.ssd >= kwargs["min"]]
        self.data = df.reset_index(drop=True)
        return self

    def ecc(self, is_ecc: bool = None):
        if is_ecc is None:
            print(
                "HetznerAuction.is_ecc() expects a positional argument of True or False"
            )
            return self
        df = self.data
        if is_ecc:
            df = df[df.is_ecc == 1]
        else:
            df = df[df.is_ecc == 0]
        self.data = df.reset_index(drop=True)
        return self

    def sort(self, how: str = "value_asc"):
        """DataFrame Sort Method

        Parameters:
          - how (str): one of the following: value_asc, value_desc, price_asc, price_desc
        """
        if how not in ["value_asc", "value_desc", "price_asc", "price_desc"]:
            how = "value_asc"
        df = self.data
        if how == "value_asc":
            df = df.sort_values("value")
        elif how == "value_desc":
            df = df.sort_values("value", ascending=False)
        elif how == "price_asc":
            df = df.sort_values("price")
        elif how == "price_desc":
            df = df.sort_values("price", ascending=False)
        self.data = df.reset_index(drop=True)
        return self

    def __str__(self):
        return str(self.data)
