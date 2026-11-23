from __future__ import annotations
import math
import numpy as np
from srpaper import runtime as rt

DATASETS = ["Set5", "Set14", "BSD100", "Urban100", "Manga109", "DIV2K-valid"]


def paired(seed, mode, comparator, dataset="Urban100"):
    serial = rt.read_csv(rt.reference_file(f"serial_x4_seed{seed}",dataset,mode))
    control = rt.read_csv(rt.reference_file(f"{comparator}_x4_seed{seed}",dataset,mode))
    if [r["image"] for r in serial] != [r["image"] for r in control]:
        raise ValueError("Unmatched image IDs")
    return serial, control


def differences(seed, mode, comparator):
    serial, control = paired(seed, mode, comparator)
    return np.array([float(a["psnr"])-float(b["psnr"]) for a,b in zip(serial,control)],dtype=np.float64)


def holm(p):
    order=np.argsort(p)
    adjusted=np.empty(len(p),dtype=np.float64)
    running=0.0
    for rank,index in enumerate(order):
        running=max(running,min(1.0,float(p[index])*(len(p)-rank)))
        adjusted[index]=running
    return adjusted


def audit():
    frozen=rt.read_json(rt.ROOT/"results/reference/completed_p1_analysis.json")
    matched=[]
    max_error=0.0
    for item in frozen["matched_comparisons"]:
        seed=int(item["training_seed"]);mode=item["inference"];dataset=item["dataset"]
        row={"seed":seed,"dataset":dataset,"inference":mode}
        for comparator,token in [("p05","parallel05"),("p1","parallel1")]:
            a,b=paired(seed,mode,comparator,dataset)
            for field in ("psnr","ssim"):
                delta=rt.mean(a,field)-rt.mean(b,field)
                expected=float(item[f"serial_minus_{token}_{field}"])
                max_error=max(max_error,abs(delta-expected))
                row[f"serial_minus_{comparator}_{field}"]=delta
                if delta<=0:
                    raise ValueError("Unexpected nonpositive matched mean")
        matched.append(row)
    if len(matched)!=24 or max_error>1e-10:
        raise ValueError("Final matched records differ")
    statistics=frozen["urban100_statistics"]
    corrected=holm([float(x["sign_flip_p_raw"]) for x in statistics])
    for item,p in zip(statistics,corrected):
        comparator="p05" if item["comparator"]=="parallel05" else "p1"
        delta=differences(int(item["seed"]),item["inference"],comparator)
        if len(delta)!=100 or abs(float(delta.mean())-item["mean_difference_db"])>1e-10:
            raise ValueError("Urban100 paired mean differs")
        if abs(float(p)-item["holm_joint8"])>1e-12:
            raise ValueError("Joint Holm correction differs")
    return {"matched_conditions_per_comparator":24,"comparators":2,"quality_means_positive":True,
            "maximum_mean_difference_from_frozen_json":max_error,"urban100_tests":8,
            "joint_holm_below_005":int(sum(corrected<0.05)),"matched_rows":matched}


def bootstrap_and_flip(values,rng,bootstrap=200000,flips=500000):
    means=[]
    for start in range(0,bootstrap,5000):
        indices=rng.integers(0,len(values),size=(min(5000,bootstrap-start),len(values)))
        means.append(values[indices].mean(axis=1))
    lower,upper=np.quantile(np.concatenate(means),[0.025,0.975])
    extreme=0
    for start in range(0,flips,5000):
        bits=rng.integers(0,2,size=(min(5000,flips-start),len(values)),dtype=np.int8)
        randomized=((bits.astype(np.float64)*2-1)*values).mean(axis=1)
        extreme+=np.count_nonzero(np.abs(randomized)>=abs(values.mean())-1e-15)
    return float(lower),float(upper),float((extreme+1)/(flips+1))


def run(args):
    out=rt.new_run("analysis")
    report=audit()
    rt.write_csv(out/"matched_comparisons.csv",report.pop("matched_rows"))
    frozen=rt.read_json(rt.ROOT/"results/reference/completed_p1_analysis.json")
    rt.write_csv(out/"archived_urban100_statistics.csv",frozen["urban100_statistics"])
    if args.resample:
        rows=[]
        for seed in (0,1):
            for mode in ("Standard","TLC"):
                for comparator in ("p05","p1"):
                    values=differences(seed,mode,comparator)
                    lo,hi,p=bootstrap_and_flip(values,np.random.default_rng(20260728))
                    rows.append({"seed":seed,"inference":mode,"comparator":comparator,"mean":float(values.mean()),
                                 "bootstrap_ci_lower":lo,"bootstrap_ci_upper":hi,"sign_flip_p_raw":p,
                                 "wins":int(sum(values>1e-12)),"losses":int(sum(values< -1e-12)),"ties":int(sum(abs(values)<=1e-12))})
        for row,p in zip(rows,holm([x["sign_flip_p_raw"] for x in rows])):
            row["holm_joint8"]=float(p)
        rt.write_csv(out/"recomputed_urban100_statistics.csv",rows)
        report["resampling"]={"bootstrap":200000,"sign_flip":500000,"rng":20260728,"stream":"reset per condition", "archived_values_preserved":True}
    rt.write_json(out/"audit.json",report)
    print(report)
    print("Results:",out)
