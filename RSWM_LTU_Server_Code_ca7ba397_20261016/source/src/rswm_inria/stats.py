"""Paired, city-stratified source-tile sensitivity intervals (NOT global city CIs)."""
import numpy as np
from .geometry import rng_for


def paired_city_summary(expected, rows_a, rows_b, bootstrap=2000, seed=0):
    """Rows must already aggregate origins/tasks/seeds to one value per source world.

    Expected rows: {world_id, city}. Result rows: {world_id, value} (None allowed).
    Missing rows and None are retained in coverage, not silently imputed to zero.
    """
    if type(bootstrap) is not int or bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    ids=[r["world_id"] for r in expected]
    if len(ids) != len(set(ids)) or not ids: raise ValueError("Duplicate or empty expected plan.")
    def index(rows):
        result={}
        for row in rows:
            key=row["world_id"];value=row["value"]
            if key not in ids or key in result: raise ValueError("Unknown or duplicate world.")
            if value is not None and not np.isfinite(value): raise ValueError("Values must be finite or None.")
            result[key]=value
        return result
    a,b=index(rows_a),index(rows_b);cities=sorted({r["city"] for r in expected});groups={}
    for city in cities:
        items=[r["world_id"] for r in expected if r["city"] == city]
        values=[None if a.get(k) is None or b.get(k) is None else float(a[k]-b[k]) for k in items]
        present=[v for v in values if v is not None]
        groups[city]={"planned_worlds":len(items),"paired_worlds":len(present),
                      "paired_coverage":len(present)/len(items),"conditional_mean_a_minus_b":float(np.mean(present)) if present else None,
                      "values":values}
    means=[g["conditional_mean_a_minus_b"] for g in groups.values()]
    mean=float(np.mean(means)) if all(v is not None for v in means) else None
    rng=rng_for("bootstrap",seed);replicates=[]
    for _ in range(bootstrap):
        city_means=[]
        for g in groups.values():
            picked=[g["values"][i] for i in rng.integers(len(g["values"]),size=len(g["values"]))]
            measured=[v for v in picked if v is not None]
            if not measured: break
            city_means.append(float(np.mean(measured)))
        if len(city_means)==len(cities):replicates.append(float(np.mean(city_means)))
    return {"cities":{c:{k:v for k,v in g.items() if k != "values"} for c,g in groups.items()},
            "equal_city_mean_a_minus_b":mean,
            "source_tile_bootstrap_95_interval":np.quantile(replicates,[.025,.975]).tolist() if replicates else None,
            "planned_bootstrap_replicates":bootstrap,"valid_bootstrap_replicates":len(replicates),
            "scope":"conditional on these cities; source-tile independence unverified without georeferencing; exploratory interval"}
