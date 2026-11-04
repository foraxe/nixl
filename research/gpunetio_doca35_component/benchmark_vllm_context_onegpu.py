"""Matched vLLM Context packed-row component benchmark; one pinned owner, no model."""
import argparse, importlib.util, json, hashlib, statistics
from pathlib import Path
import torch
p=argparse.ArgumentParser(); p.add_argument('--library',required=True); p.add_argument('--module',required=True); p.add_argument('--port',type=int,required=True); p.add_argument('--out',required=True); a=p.parse_args()
spec=importlib.util.spec_from_file_location('context_module',a.module); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.set_num_threads(2); rows=8192; width=264; maximum=6144
base=torch.arange(rows*width,dtype=torch.int64).remainder(251).to(torch.uint8).view(rows,width)
torch.cuda.set_device(0); table=torch.empty_like(base,pin_memory=True); table.copy_(base)
owner=m.Context(a.library,0,a.port); owner.register([table],host=True)
reader=None
try:
 torch.cuda.set_device(0); arena=torch.full((maximum*width+256,),165,dtype=torch.uint8,device='cuda'); stats=torch.zeros(8,dtype=torch.uint64,device='cuda'); scratch=maximum*width+128
 reader=m.Context(a.library,0,a.port+1); reader.register([arena],host=False); reader.connect_raw(owner.metadata(),list(owner.addresses),list(owner.lengths),0)
 gen=torch.Generator().manual_seed(20261103); results=[]
 for n in (384,6144):
  ids_cpu=torch.randint(rows,(n,),generator=gen,dtype=torch.int64); ids=ids_cpu.cuda(); expected=base[ids_cpu]
  ev0,ev1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  def execute():
   reader.fetch(ids,n,0,0,scratch,512,stats,weight_index=0,scale_index=0,row_lo=0,row_hi=rows); reader.record_consumed()
  for _ in range(3): execute()
  torch.cuda.synchronize(); assert int(stats[0])==0 and int(stats[4])==n
  assert torch.equal(arena[:n*width].view(n,width).cpu(),expected)
  samples=[]
  for _ in range(7):
   ev0.record(); execute(); ev1.record(); ev1.synchronize(); samples.append(ev0.elapsed_time(ev1)*1000)
  assert int(stats[0])==0 and int(stats[4])==n and torch.equal(arena[:n*width].view(n,width).cpu(),expected)
  results.append({'queries':n,'gpu_event_us':samples,'median_us':statistics.median(samples),'validation':'PASS'})
 reader.close();reader=None;torch.cuda.set_device(0);owner.close();owner=None
 out={'library_sha256':hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),'table_rows':rows,'samples':7,'warmups':3,'results':results}
 Path(a.out).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out))
finally:
 if reader is not None: torch.cuda.set_device(0); reader.close()
 if owner is not None: torch.cuda.set_device(0); owner.close()
