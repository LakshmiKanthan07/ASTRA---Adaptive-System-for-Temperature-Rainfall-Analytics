"""
from ecmwf.opendata import Client

client = Client(source="ecmwf")

client.retrieve(
    date="2026-09-04",
    time=0,
    stream="oper",
    type="fc",
    step=24,
    param=["2t", "10u", "10v", "msl", "tp"],
    target="hres.grib2"
)"""

from ecmwf.opendata import Client

client = Client(source="ecmwf")

client.retrieve(
    date="2026-09-04",
    time=0,
    stream="enfo",
    type="pf",
    step=24,
    param=["2t", "10u", "10v", "msl", "tp"],
    target="ens.grib2"
)

