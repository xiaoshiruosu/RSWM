"""Run synthetic mathematical controls, not remote-sensing model experiments."""
import json
import argparse
import time
from pathlib import Path
from rswm_inria.geometry import rng_for
from rswm_inria.tasks import packages,interpolate,target_labels
from rswm_inria.baselines import CopyModel,RandomModel,StaticNoiseMap,OracleMap,WindowMap
from rswm_inria.runner import run_task,verify_journal


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--origins",nargs="+",type=int,default=[0,1])
    parser.add_argument("--families",nargs="+",default=["F2","F3","F5","F7"])
    parser.add_argument("--out",default="evidence/diagnostics.json")
    args=parser.parse_args()
    world=rng_for("diagnostic-suite-world").integers(0,256,(2048,2048,3),dtype="uint8")
    records=[];t0=time.perf_counter()
    for origin in args.origins:
        for task,labels in packages("diagnostic-suite","diagnostic_only",origin):
            if task["family"] not in args.families:continue
            factories={"copy":CopyModel,"random":RandomModel,"fixed_noise_map":StaticNoiseMap,
                       "oracle":lambda:OracleMap(world,task["origin"]),
                       "eager_diagnostic_map":lambda:WindowMap(eager=True),
                       "lazy_diagnostic_map":lambda:WindowMap(eager=False)}
            for name,factory in factories.items():
                result,journal=run_task(task,world,factory(),seed=0,labels=labels)
                verify_journal(journal.rows)
                result["model"]=name;result["origin_index"]=origin
                reports=[result["reports"]] if task["family"]=="F7" else list(result["reports"].values())
                if name in ("oracle","fixed_noise_map","eager_diagnostic_map","lazy_diagnostic_map"):
                    for report in reports:
                        for metric in report["metrics"].values():
                            if metric["expected_pixels"]:
                                assert metric["mae"]==0 and metric["coverage"]==1
                records.append(result)
                print(origin,task["family"],name,flush=True)
    q=interpolate([(0,0),(480,0),(480,480),(-480,480),(-480,-480),(480,0)],[16,16,32,16,16])
    labels=target_labels(q,(480,0))
    result={"kind":"synthetic_mathematical_controls_only","runs":records,"run_count":len(records),
            "new_output_events":sum(r["reports"]["planned_outputs"] if r["family"]=="F7" else sum(s["planned_outputs"] for s in r["reports"].values()) for r in records),"elapsed_seconds":time.perf_counter()-t0,
            "fixed_example_reentry":[int(labels["reentry_event"].min()),int(labels["reentry_event"].max())],
            "fixed_example_gap":[int(labels["invisible_query_gap"].min()),int(labels["invisible_query_gap"].max())]}
    Path(args.out).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":main()
