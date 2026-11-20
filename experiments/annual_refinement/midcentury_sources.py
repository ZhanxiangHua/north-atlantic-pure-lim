"""Resolve the two CMIP6 release intervals used in the midcentury replay."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from annual_lim.layout import CONFIG, read
from tc_emulator.common import MODEL, ROOT, TABLES, atomic

REPORT = ROOT / "reports/annual_refinement/midcentury"
PLAN = Path(__file__).with_name("midcentury_plan.json")
ENDPOINT = "https://esgf-data.dkrz.de/esg-search/search"


def next_interval(variable):
    table = TABLES[variable]
    dates = "20550101-20741231" if table == "day" else "205501-207412"
    params = dict(project="CMIP6", type="File", source_id=MODEL, experiment_id="ssp245",
                  variable_id=variable, table_id=table, grid_label="gn", data_node="esgf3.dkrz.de",
                  query=f"title:*{dates}.nc", format="application/solr+json", limit=200,
                  fields="title,dataset_id,url,checksum,checksum_type,size,member_id", distrib="false")
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))))
        reply = session.get(ENDPOINT, params=params, timeout=60)
        reply.raise_for_status()
        release = reply.json()["response"]
    if len(release["docs"]) != release["numFound"]:
        raise ValueError("Truncated ESGF release list")
    rows = []
    for doc in release["docs"]:
        member = doc["member_id"][0]
        title = f"{variable}_{table}_{MODEL}_ssp245_{member}_gn_{dates}.nc"
        if doc["title"] != title or doc["checksum_type"] != ["SHA256"]:
            raise ValueError(f"Unexpected release {doc['title']}")
        url, = [u.split("|")[0].replace("http://", "https://") for u in doc["url"]
                if u.endswith("|HTTPServer")]
        rows.append(dict(member=member, variable=variable, table=table, experiment="ssp245",
                         version=doc["dataset_id"].split("|")[0].split(".")[-1], file=title,
                         bytes=doc["size"], checksum=doc["checksum"][0], esgf=url))
    return pd.DataFrame(rows)


def build():
    old = pd.read_csv(ROOT / "reports/mpi2050/cmip6_files.csv")
    with ThreadPoolExecutor(4) as pool:
        new = pd.concat(pool.map(next_interval, TABLES), ignore_index=True)
    roles = read(CONFIG / "members.json")
    keys = ["member", "variable"]
    wanted = {(m, v) for m in roles["test"] for v in TABLES} | {(m, "tas") for m in roles["train"]}
    new = new[new.set_index(keys).index.isin(wanted)].sort_values("version").drop_duplicates(keys, keep="last")
    for df in (old, new):
        if set(map(tuple, df[keys].values)) != wanted or len(df) != len(wanted):
            raise ValueError("Missing a member or variable")
    out = pd.concat([old.assign(first_year=2035, last_year=2054),
                     new.assign(first_year=2055, last_year=2074)], ignore_index=True)
    with atomic(REPORT / "cmip6_files.csv") as tmp:
        out.sort_values(["first_year", *keys]).to_csv(tmp, index=False)
    print(f"Pinned {len(out)} source files")


if __name__ == "__main__":
    build()
