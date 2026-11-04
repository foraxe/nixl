"""Isolate per-QP READ service: identical rows/bytes, 1/2/4 independent lanes."""
import argparse, hashlib, json, time
import ctypes as C
import base64
from owner_control import OwnerClient
from itertools import product
from pathlib import Path
import torch
from bench_native_host import make_synthetic_table, quantiles
from engram_transport import Context
from row264_kernels import gather_row264
from read_group import ReadGroup
import dedup_rows
import numpy as np
ap=argparse.ArgumentParser()
ap.add_argument('--library',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--gpu',type=int,default=7)
ap.add_argument('--port-base',type=int,default=59910)
ap.add_argument('--samples',type=int,default=12)
ap.add_argument('--windows',type=int,nargs='+',default=[512])
ap.add_argument('--lanes',type=int,nargs='+',choices=(1,2,4,8,16),default=[1,2,4])
ap.add_argument('--pending',type=int,nargs='+',choices=(0,1),default=[0])
ap.add_argument('--layouts',nargs='+',choices=('multi','single','stagger'),default=['multi'])
ap.add_argument('--profile-dir',type=Path)
ap.add_argument('--timeline',action='store_true')
ap.add_argument('--cost-trace',action='store_true')
ap.add_argument('--large-only',action='store_true')
ap.add_argument('--stagger-ns',type=int,default=0)
ap.add_argument('--remote-ready',type=Path)
ap.add_argument('--reader-rails',nargs='+')
ap.add_argument('--pipeline',type=int,nargs='+',choices=(1,2),default=[1])
ap.add_argument('--polling',nargs='+',choices=('parallel','terminal'),default=['parallel'])
ap.add_argument('--dedup',type=int,nargs='+',choices=(0,1),default=[0])
ap.add_argument('--ids-file',type=Path)
ap.add_argument('--ids-key')
ap.add_argument('--rank',type=int,choices=(0,1,2,3),default=1)
ap.add_argument('--adaptive',action='store_true')
ap.add_argument('--cuda-graph',action='store_true')
ap.add_argument('--compact-unique',action='store_true')
ap.add_argument('--replay-suite',type=Path)
args=ap.parse_args()
if args.out.exists():raise FileExistsError(args.out)
capacity=max(args.lanes)
if not 1 <= args.port_base <= 65535-capacity:ap.error('port range does not fit')
if args.reader_rails and len(args.reader_rails)!=capacity:ap.error('one NIC name per allocated QP required')
if 2 in args.pipeline and 'multi' in args.layouts:ap.error('pipeline requires grouped kernel')
torch.set_num_threads(4);torch.cuda.set_device(args.gpu)
external_ids=torch.from_numpy(np.load(args.ids_file)[args.ids_key].copy()) if args.ids_file and args.ids_key else None
rows=1<<18; maximum=max(49152,len(external_ids) if external_ids is not None else 0)
w,s=make_synthetic_table(rows)
table=torch.empty((rows,264),dtype=torch.uint8,pin_memory=True)
table[:,:256].copy_(w);table[:,256:].copy_(s)
owner=None;lease=None
arena=torch.full((64+maximum*264+256,),0xa5,dtype=torch.uint8,device='cuda')
readers=[];streams=[];events=[]
stats=torch.zeros((capacity,8),dtype=torch.uint64,device='cuda')
traces=torch.zeros((capacity,520),dtype=torch.uint64,device='cuda') if args.timeline else None
costs=torch.zeros((capacity,4),dtype=torch.uint64,device='cuda') if args.cost_trace else None
groups={}
setup_checkpoints=[]
def process_state():
 status=Path('/proc/self/status').read_text().splitlines()
 return {line.split(':',1)[0]:line.split(':',1)[1].strip() for line in status if line.startswith(('VmRSS:','Threads:'))}
setup_before=process_state();free_before=torch.cuda.mem_get_info()[0];setup_start=time.perf_counter()
try:
 if args.remote_ready:
  ready=json.loads(args.remote_ready.read_text());client=OwnerClient(ready['host'],ready['port'],ready['checkpoint_id'],ready['generation'])
  lease=client.attach();remote=lease.metadata['rails'][__import__('os').environ['GPUNETIO_DEVICE_API_RDMA']]
  if lease.metadata['rows']!=rows:raise ValueError('remote synthetic oracle shape mismatch')
  expected_table_sha=hashlib.sha256(memoryview(table.numpy())).hexdigest()
  if lease.metadata.get('table_sha256')!=expected_table_sha:raise ValueError('remote synthetic bytes differ from local oracle')
 else:
  owner=Context(args.library,args.gpu,args.port_base);owner.register([table],host=True)
 initial_device=__import__('os').environ['GPUNETIO_DEVICE_API_RDMA']
 for i in range(capacity):
  if args.reader_rails:
   if not lease:raise ValueError('multi-NIC fixture requires remote owner')
   __import__('os').environ['GPUNETIO_DEVICE_API_RDMA']=args.reader_rails[i]
   remote=lease.metadata['rails'][args.reader_rails[i]]
  c=Context(args.library,args.gpu,args.port_base+1+i);readers.append(c)
  c.register([arena],host=False)
  if lease:
   c.connect_raw(base64.b64decode(remote['native_md']),remote['addresses'],remote['lengths'],remote['owner_gpu'])
  else:c.connect(owner)
  if traces is not None:
   c.library.engram_set_trace.argtypes=[C.c_void_p,C.c_uint64]
   c.check(c.library.engram_set_trace(c.handle,traces[i].data_ptr()))
  if costs is not None:
   c.library.engram_set_cost_trace.argtypes=[C.c_void_p,C.c_uint64]
   c.check(c.library.engram_set_cost_trace(c.handle,costs[i].data_ptr()))
  streams.append(torch.cuda.Stream());events.append(torch.cuda.Event())
  if i+1 in args.lanes:
   setup_checkpoints.append(dict(contexts=i+1,seconds=time.perf_counter()-setup_start,
     process=process_state(),gpu_free_delta_bytes=free_before-torch.cuda.mem_get_info()[0]))
 __import__('os').environ['GPUNETIO_DEVICE_API_RDMA']=initial_device
 for layout in args.layouts:
  if layout=='multi':continue
  for n in sorted(set(args.lanes+([1] if args.adaptive else []))):
   for depth in args.pipeline:
    for polling in args.polling:
     group=ReadGroup(readers[:n]);groups[(n,layout,depth,polling)]=group
     group.set_pipeline(depth);group.set_terminal(polling=='terminal')
     if layout=='stagger':
      group.lib.engram_read_group_stagger.argtypes=[C.c_void_p,C.c_uint64]
      readers[0].check(group.lib.engram_read_group_stagger(group.handle,args.stagger_ns))
 setup=dict(seconds=time.perf_counter()-setup_start,before=setup_before,after=process_state(),gpu_free_delta_bytes=free_before-torch.cuda.mem_get_info()[0],allocated_contexts=capacity,checkpoints=setup_checkpoints)
 results=[]
 if args.replay_suite:
  archive=np.load(args.ids_file)
  items=[(item['key'],item['rank'],torch.from_numpy(archive[item['key']].copy()),len(archive[item['key']]),True) for item in json.loads(args.replay_suite.read_text())]
 else:items=[('component',args.rank,external_ids,n,private) for n,private in ([(len(external_ids),True)] if external_ids is not None else [(49152,True)] if args.large_only else [(24,False),(384,True),(6144,False),(49152,True)])]
 for case_label,case_rank,external_ids,n,private in items:
  if n>maximum:raise ValueError('replay exceeds fixed scratch')
  lo,hi=(rows*case_rank//4,rows*(case_rank+1)//4) if private else (0,rows)
  cpu=external_ids if external_ids is not None else torch.randint(rows,(n,),generator=torch.Generator().manual_seed(n+20261027))
  if external_ids is None:cpu[:8]=torch.tensor([0,1,lo,lo+1,hi-1,min(hi,rows-1),lo,lo])
  ids=cpu.cuda();mask=(cpu>=lo)&(cpu<hi)
  expected=(w[cpu].view(torch.float8_e4m3fn).float()*torch.exp2(s[cpu].float()-127).repeat_interleave(32,1)).to(torch.bfloat16).masked_fill(~mask[:,None],0)
  out=torch.empty((n,256),device='cuda',dtype=torch.bfloat16)
  cases=list(product(args.lanes,args.windows,args.pending,args.layouts,args.pipeline,args.polling,args.dedup))
  samples={k:[] for k in cases}
  scratch,reps=dedup_rows.workspace(n,'cuda')
  unique_queries=torch.empty_like(reps);unique_count=scratch[-1:]
  def run(lanes,window,pending,layout,depth,polling,dedup):
   lanes=1 if args.adaptive and n<4096 else lanes
   if args.adaptive and n<4096:dedup=0
   if dedup:dedup_rows.prepare(ids,lo,hi,scratch,reps,unique_queries if args.compact_unique else None)
   if layout!="multi":
    groups[(lanes,layout,depth,polling)].fetch(ids,n,64,len(arena)-64,0,window,stats[:lanes],lo,hi,bool(pending),reps if dedup else None,unique_queries if dedup and args.compact_unique else None,unique_count if dedup and args.compact_unique else None)
    gather_row264(arena[64:].data_ptr(),ids,out,lo,hi,query_order=True,representatives=reps if dedup else None)
    groups[(lanes,layout,depth,polling)].record_consumed()
    return
   main=torch.cuda.current_stream();chunk=(n+lanes-1)//lanes
   for i in range(lanes):
    a=i*chunk;b=min(n,a+chunk)
    streams[i].wait_stream(main)
    with torch.cuda.stream(streams[i]):
     readers[i].fetch(ids[a:b],b-a,64+a*264,0,len(arena)-64+i,window,stats[i],weight_index=0,scale_index=0,row_lo=lo,row_hi=hi,row264=True,pending=bool(pending))
     events[i].record()
   for i in range(lanes):main.wait_event(events[i])
   gather_row264(arena[64:].data_ptr(),ids,out,lo,hi,query_order=True,representatives=reps if dedup else None)
   for i in range(lanes):
    streams[i].wait_stream(main)
    with torch.cuda.stream(streams[i]):readers[i].record_consumed()
  def check(lanes,dedup):
   lanes=1 if args.adaptive and n<4096 else lanes
   if args.adaptive and n<4096:dedup=0
   assert torch.equal(out.cpu().view(torch.int16),expected.view(torch.int16))
   st=stats[:lanes].cpu();expected_ops=len(torch.unique(cpu[mask])) if dedup else int(mask.sum())
   assert (st[:,0]==0).all() and int(st[:,4].sum())==expected_ops
   raw=arena.cpu();got=raw[64:64+n*264].view(n,264)
   fetched=mask
   if dedup:
    r=reps.cpu();fetched=mask&(r==torch.arange(n))
    assert torch.equal(got[r[mask]],table[cpu[mask]])
   assert torch.equal(got[fetched],table[cpu[fetched]])
   assert (got[~fetched]==0xa5).all() and (raw[:64]==0xa5).all() and (raw[64+n*264:]==0xa5).all()
   return st.tolist()
  prepared_graphs={};capture_stream=torch.cuda.Stream() if args.cuda_graph else None
  for case in cases:
   if args.cuda_graph:
    with torch.cuda.stream(capture_stream):
     for _ in range(3):arena.fill_(0xa5);run(*case)
   else:
    for _ in range(3):arena.fill_(0xa5);run(*case)
   torch.cuda.synchronize();check(case[0],case[-1])
   if args.cuda_graph:
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=capture_stream):run(*case)
    prepared_graphs[case]=graph
  for t in range(args.samples):
   for k in (cases if t%2==0 else cases[::-1]):
    arena.fill_(0xa5);torch.cuda.synchronize()
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();host_start=time.perf_counter_ns()
    if args.cuda_graph:prepared_graphs[k].replay()
    else:run(*k)
    host_us=(time.perf_counter_ns()-host_start)/1000;b.record();b.synchronize();st=check(k[0],k[-1])
    samples[k].append(dict(host_submit_us=host_us,gpu_us=a.elapsed_time(b)*1000,fetch_max_us=max(x[3] for x in st)/1000,read_wqes=sum(x[4] for x in st),doorbells=sum(x[6] for x in st),post_sum_us=sum(x[1] for x in st)/1000,wait_max_us=max(x[2] for x in st)/1000))
    if costs is not None:samples[k][-1]['cost_ns']=costs[:k[0]].cpu().tolist()
    if traces is not None:
     active=1 if args.adaptive and n<4096 else k[0]
     raw=traces[:active].cpu().tolist()
     samples[k][-1]['timeline']=[row[:8+4*min(128,row[7])] for row in raw]
     active_posts=[row[1] for row in raw if row[1]]
     samples[k][-1].update(
      completion_window_us=(max(row[3] for row in raw)-min(active_posts))/1000 if active_posts else 0,
      first_submit_spread_us=(max(active_posts)-min(active_posts))/1000 if active_posts else 0)
  for k,vals in samples.items():results.append(dict(case=case_label,rank=case_rank,lanes=k[0],window=k[1],pending=k[2],layout=k[3],pipeline=k[4],polling=k[5],dedup=k[6],effective_dedup=0 if args.adaptive and n<4096 else k[6],active_lanes=1 if args.adaptive and n<4096 else k[0],queries=n,owned=int(mask.sum()),gpu_us=quantiles([x['gpu_us'] for x in vals]),fetch_max_us=quantiles([x['fetch_max_us'] for x in vals]),samples=vals,completion_window_us=quantiles([v['completion_window_us'] for v in vals]) if traces is not None else None))
 if args.profile_dir:
  args.profile_dir.mkdir(exist_ok=False,parents=True)
  for layout in args.layouts:
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    run(args.lanes[-1],args.windows[-1],args.pending[-1],layout,args.pipeline[-1],args.polling[-1],args.dedup[-1])
    torch.cuda.synchronize()
   prof.export_chrome_trace(str(args.profile_dir/(layout+'.json')))
finally:
 for group in groups.values():group.close()
 for c in readers:c.close()
 if owner:owner.close()
 if lease:lease.release()
result=dict(validation='PASS',compact_unique=args.compact_unique,cuda_graph=args.cuda_graph,wait_stat_definition='FIFO build: actual poll-entry through CQ retirement; batch post-to-done latency remains in timeline',setup=setup,remote_owner=bool(args.remote_ready),reader_rails=args.reader_rails,rdma_device=__import__('os').environ.get('GPUNETIO_DEVICE_API_RDMA'),cpu_affinity=sorted(__import__('os').sched_getaffinity(0)),timeline=args.timeline,stagger_ns=args.stagger_ns,scope='component only; same queries, bytes, row layout, ownership, per-WQE signaling; vary specified QPs/streams and input-window size',library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),results=results)
args.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!="results"}))
for x in results:print(json.dumps({k:v for k,v in x.items() if k!="samples"}))
