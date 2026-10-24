"""Selected NIC8/G4, block0, two warmups+one measured request per mode."""
import argparse
import json
import sqlite3
from pathlib import Path


def analyze(path):
    c=sqlite3.connect(path)
    sql="""select k.start,k.end,k.graphNodeId,k.correlationId,k.streamId,s.value
           from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id
           where k.graphNodeId>0 order by k.start"""
    rows=c.execute(sql).fetchall()
    correlations=sorted({r[3] for r in rows},key=lambda key:min(r[0] for r in rows if r[3]==key))
    assert len(correlations)==6,len(correlations)
    out=[]
    for i,key in enumerate(correlations):
        data=[r for r in rows if r[3]==key]
        actors=sorted((r for r in data if r[5]=="issue"),key=lambda r:r[2])
        gemms=sorted((r for r in data if r[5].startswith("nvjet_")),key=lambda r:r[0])
        assert len(actors)==16 and len(gemms)==8,(len(actors),len(gemms))
        a0,a1=actors[:8],actors[8:];g0=gemms[:4]
        ready0=max(r[1] for r in a0);end0=max(r[1] for r in g0)
        out.append(dict(mode="SERIAL" if i<3 else "LOOKAHEAD",replay=i%3,correlation=key,
                        actor1_start_delays_ms=[(r[0]-ready0)/1e6 for r in a1],
                        actor1_duration_ms=[(r[1]-r[0])/1e6 for r in a1],
                        actor1_after_compute0=sum(r[0]>=end0 for r in a1),
                        compute0_kernel_sum_ms=sum(r[1]-r[0] for r in g0)/1e6,
                        compute0_span_ms=(max(r[1] for r in g0)-min(r[0] for r in g0))/1e6,
                        actor1_streams=[r[4] for r in a1],compute0_streams=sorted({r[4] for r in g0})))
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("input",type=Path)
    a=p.parse_args();print(json.dumps(analyze(a.input),indent=2))
