import argparse
import json
from pathlib import Path
import statistics as S


def analyze(path, blocks):
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.startswith("{")]
    assert any(r.get("kind")=="complete" and r.get("status")=="PASS" for r in rows),"incomplete run"
    assert any(r.get("kind")=="negative_validation" and r["detected_mismatches"]>0 for r in rows)
    cfg=next(r for r in rows if r["kind"]=="config")
    data=[r for r in rows if r["kind"]=="result"]
    assert all(r["mismatch"]==0 for r in data)
    groups={key:[r for r in data if (r["arm"],r["mode"],r["gemms"])==key]
            for key in {(r["arm"],r["mode"],r["gemms"]) for r in data}}
    assert all(sorted(r["block"] for r in a)==list(range(blocks)) for a in groups.values())
    metrics=["e2e_ms","copy0_ms","copy1_ms","compute0_ms","compute1_ms","wait0_ms","wait1_ms"]
    summary=[dict(arm=k[0],mode=k[1],gemms=k[2],**{m:S.median(r[m] for r in a) for m in metrics})
             for k,a in sorted(groups.items())]
    comparisons=[]
    for arm,g in sorted({(r["arm"],r["gemms"]) for r in data}):
        serial=groups.get((arm,"SERIAL",g));pipe=groups.get((arm,"LOOKAHEAD",g))
        if not serial or not pipe:continue
        ratios=[next(r["e2e_ms"] for r in serial if r["block"]==b)/
                next(r["e2e_ms"] for r in pipe if r["block"]==b) for b in range(blocks)]
        comparisons.append(dict(arm=arm,gemms=g,serial_over_pipeline=S.median(ratios),
                                favorable_blocks=sum(x>1 for x in ratios),blocks=blocks))
    best=[]
    for g in sorted({r["gemms"] for r in data}):
        entry={"gemms":g}
        for family,prefix in [("CE","CE_"),("SM","SM_"),("NIC","NIC_")]:
            candidates=[r for r in summary if r["gemms"]==g and r["arm"].startswith(prefix)]
            if candidates:
                r=min(candidates,key=lambda r:r["e2e_ms"])
                entry[family]={k:r[k] for k in ["arm","mode","e2e_ms","wait1_ms","compute0_ms","compute1_ms"]}
        best.append(entry)
    return dict(config=cfg,summary=summary,overlap=comparisons,best=best)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("input",type=Path);p.add_argument("--blocks",type=int,default=7)
    a=p.parse_args();print(json.dumps(analyze(a.input,a.blocks),indent=2))
