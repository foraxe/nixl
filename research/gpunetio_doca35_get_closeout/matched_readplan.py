#!/usr/bin/env python3
"""Cross-node or loopback synthetic row264 benchmark using frozen ReadPlan."""
import argparse
import base64
import hashlib
import gc
import json
import os
import socket
import statistics
import sys
import threading
import time
import math
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
root = HERE / 'baseline' / 'python' / 'sglang'
# Do not import an installed SGLang package or its optional model dependencies.
for name, path in [('sglang',root), ('sglang.srt',root/'srt'),
                   ('sglang.srt.engram_share',root/'srt'/'engram_share'),
                   ('sglang.srt.engram_share.rdma',root/'srt'/'engram_share'/'rdma')]:
    module=types.ModuleType(name); module.__path__=[str(path)]; sys.modules[name]=module
from sglang.srt.engram_share.rdma.engram_transport import Context
from sglang.srt.engram_share.rdma.read_plan import ReadPlan
from sglang.srt.engram_share.rdma.row264_kernels import gather_row264

p=argparse.ArgumentParser()
p.add_argument('--role',choices=('owner','reader','loopback'),default='loopback')
p.add_argument('--library',required=True)
p.add_argument('--mode',choices=('eager','graph'),default='eager')
p.add_argument('--arm',default='unspecified')
p.add_argument('--port',type=int,required=True,help='Native owner or reader port base')
p.add_argument('--control-port',type=int,help='Cross-node control TCP port')
p.add_argument('--owner-address',help='Reader connects to this address')
p.add_argument('--listen-address',default='0.0.0.0')
p.add_argument('--out')
p.add_argument('--rows',type=int,default=65536)
p.add_argument('--sizes',type=int,nargs='+',default=[2,64,1024,4096,32768])
p.add_argument('--plan',choices=('delivered','multirail'),default='delivered')
p.add_argument('--dispatch',choices=('delivered','optimized'),default='delivered')
p.add_argument('--rails',nargs='+')
p.add_argument('--weights',type=int,nargs='+')
p.add_argument('--prequeue-cycles',type=int,default=40000000)
p.add_argument('--port-burst',type=int,default=0,help='Untimed final-size repetitions for node-wide NIC counters')
p.add_argument('--decode-overlap',action='store_true',help='Run independent first-rail decode graph with prefetch')
p.add_argument('--input-pattern',choices=('unique','paired'),default='unique')
p.add_argument('--fault-probe',choices=('negative','upper'),help='Disposable reader: expect status 2 and process-exit quarantine')
a=p.parse_args()
if min(a.sizes)<=0 or a.rows<max(a.sizes) or a.rows>2**32: p.error('rows must cover sizes and fit 32-bit row encoding')
if math.gcd(a.rows,104729)!=1: p.error('rows must be coprime with 104729 for unique IDs')
if a.role!='loopback' and not a.control_port: p.error('cross-node roles require --control-port')
if a.port_burst<0: p.error('--port-burst must be nonnegative')
if a.decode_overlap and (a.role=='owner' or a.plan!='multirail' or a.mode!='graph' or a.port>65495): p.error('--decode-overlap requires multirail reader graph and port <= 65495')
if a.fault_probe and a.role!='reader': p.error('fault probe requires a separate cross-node reader process')
if a.role=='reader' and not a.owner_address: p.error('reader requires --owner-address')
if a.role!='owner' and not a.out: p.error('reader requires --out')
if a.dispatch=='optimized' and a.plan!='multirail': p.error('optimized dispatch requires multirail plan')
if a.plan=='multirail' and (not a.rails or len(a.rails)!=4): p.error('multirail requires four --rails')
if a.weights and (len(a.weights)!=4 or min(a.weights)<0 or sum(a.weights)==0): p.error('weights must be four nonnegative integers')
torch.cuda.set_device(0)

def tables_for_owner():
    tables=[]; row_ids=torch.arange(a.rows,dtype=torch.int64)
    for li in range(2):
        t=torch.empty((a.rows,264),dtype=torch.uint8,pin_memory=True); t[:,:256]=0x38
        for bit in range(32): t[:,bit]=0x38+(((row_ids >> bit)&1).to(torch.uint8)*8)
        t[:,32]=0x38+li*8
        t[:,256:]=127; tables.append(t)
    return tables

def make_owner():
    owner=Context(a.library,0,a.port); owner.register(tables_for_owner(),True)
    return owner, {'native_md':base64.b64encode(owner.metadata()).decode(), 'addresses':list(owner.addresses), 'lengths':list(owner.lengths), 'owner_gpu':0, 'rows':a.rows}

def loaded_images():
    images={}
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts=line.split(maxsplit=5)
        if len(parts)<6 or not parts[5].startswith('/'):
            continue
        mapped=parts[5]; deleted=mapped.endswith(' (deleted)')
        path=mapped[:-10] if deleted else mapped
        name=Path(path).name
        if not (name.startswith(('libnixl.so','libplugin_GPUNETIO.so','libdoca', 'libibverbs.so','libmlx5.so')) and '.so' in name):
            continue
        if mapped in images: continue
        entry={'path':path,'deleted':deleted,'mapped_device':parts[3],'mapped_inode':int(parts[4])}
        try:
            if deleted: raise FileNotFoundError('mapped file was deleted; pathname cannot identify mapped inode')
            stat=os.stat(path); devmajor,devminor=(int(x,16) for x in parts[3].split(':'))
            if (stat.st_ino!=int(parts[4]) or os.major(stat.st_dev)!=devmajor or os.minor(stat.st_dev)!=devminor):
                raise OSError('mapped inode or device differs from path; cannot identify loaded image')
            with open(path,'rb') as f:
                digest=hashlib.sha256()
                for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
            entry['sha256']=digest.hexdigest()
        except OSError as exc: entry['error']=str(exc)
        images[mapped]=entry
    return list(images.values())

def send(sock,obj): sock.sendall(json.dumps(obj,separators=(',',':')).encode()+b'\n')
def recv(sock):
    data=bytearray()
    while len(data)<1024*1024:
        part=sock.recv(1)
        if not part: raise EOFError('control connection closed')
        if part==b'\n': return json.loads(data)
        data.extend(part)
    raise ValueError('control message too large')

if a.role=='owner':
    owner,metadata=make_owner()
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind((a.listen_address,a.control_port)); listener.listen(1)
        print('OWNER_READY',flush=True)
        conn,_=listener.accept()
        with conn:
            send(conn,metadata)
            try: msg=recv(conn)
            except Exception as exc: msg={'error':str(exc)}
            if msg.get('status')!='drained':
                print('OWNER_RETAINED '+json.dumps(msg),flush=True)
                threading.Event().wait()
            owner.close(); send(conn,{'status':'closed'}); print('OWNER_CLOSED',flush=True)
    sys.exit(0)

owner=None; conn=None; plan=None; success=False
try:
    if a.role=='loopback': owner,md=make_owner()
    else:
        conn=socket.create_connection((a.owner_address,a.control_port),timeout=60); conn.settimeout(None); md=recv(conn)
    if md['rows']!=a.rows: raise ValueError('owner/reader row count mismatch')
    maximum=max(a.sizes)
    if a.plan=='multirail':
        from candidate_readplan import MultiRailReadPlan
        plan=MultiRailReadPlan(a.library,0,a.port,16,rails=a.rails,weights=a.weights,dispatch=a.dispatch,max_queries=maximum,graph_sizes=tuple(a.sizes) if a.mode=='graph' else ())
    else:
        plan=ReadPlan(a.library,0,a.port+(100 if owner else 0),16,maximum,tuple(a.sizes) if a.mode=='graph' else ())
    p_off=64; s_off=p_off+maximum*256+64; dump=(s_off+maximum*8+127)//64*64; span=dump+64
    arena=torch.empty(span*2,dtype=torch.uint8,device='cuda'); plan.register([arena],False)
    plan.connect_raw(base64.b64decode(md['native_md']),md['addresses'],md['lengths'],md['owner_gpu'])
    stats=[torch.zeros(8,dtype=torch.uint64,device='cuda') for _ in range(2)]
    samples={}; wall_samples={}; submit_samples={}; counts_by_size={}; lane_samples={}; checks=0; stream=torch.cuda.Stream()
    if a.fault_probe:
        # Native context becomes sticky after status 2: retain all registrations
        # until process exit.  Owner is deliberately not sent a drain message.
        invalid=-1 if a.fault_probe=='negative' else a.rows
        with torch.cuda.stream(stream):
            bad_ids=torch.tensor([invalid],dtype=torch.int64,device='cuda')
            plan.fetch(bad_ids,1,p_off,0,dump,512,stats[0],weight_index=0,scale_index=0,row_lo=0,row_hi=a.rows,row264=True)
            stream.synchronize(); st=stats[0].cpu().tolist()
        result={'status':'PASS' if st[0]==2 and st[4]==0 else 'FAIL',
                'fault_probe':a.fault_probe,'id':invalid,'native_status':st,
                'cleanup':'context quarantined; reader process exit; owner retained',
                'library_sha256':hashlib.sha256(Path(a.library).read_bytes()).hexdigest()}
        Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
        print('FAULT_PROBE '+json.dumps(result),flush=True)
        conn.close()
        os._exit(0 if result['status']=='PASS' else 1)
    with torch.cuda.stream(stream):
        for n in a.sizes:
            ids=[((torch.arange(n,dtype=torch.int64,device='cuda')*104729+li*31)%a.rows).contiguous() for li in range(2)]
            outputs=[torch.empty((n,256),dtype=torch.bfloat16,device='cuda') for _ in range(2)]
            lane_stats=[torch.empty_like(plan.stats) for _ in range(2)]
            def step():
                # Prefetch both layers, then record each consumption only after its gather.
                for li in range(2):
                    plan.fetch(ids[li],n,li*span+p_off,0,li*span+dump,512,stats[li],weight_index=li,scale_index=li,row_lo=0,row_hi=a.rows,row264=True)
                    lane_stats[li].copy_(plan.stats)
                for li in range(2):
                    gather_row264(arena.data_ptr()+li*span+p_off,ids[li],outputs[li],0,a.rows,query_order=True,representatives=plan.representatives(li)); plan.record_consumed()
            step(); stream.synchronize()
            timings=[]; walls=[]; submits=[]; counts=[]; lanes=[]
            for i in range(25):
                for li in range(2):
                    index=torch.arange(n,dtype=torch.int64,device='cuda')
                    if a.input_pattern=='paired': index=torch.div(index,2,rounding_mode='floor')
                    ids[li].copy_((index*104729+li*31+(i+1)*1031)%a.rows)
                stream.synchronize() # prepare IDs before timing and prequeue delay
                start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                if a.prequeue_cycles: torch.cuda._sleep(a.prequeue_cycles)
                wall_start=time.perf_counter_ns(); start.record(stream); step(); end.record(stream); submit_us=(time.perf_counter_ns()-wall_start)/1000
                if a.prequeue_cycles and start.query(): raise RuntimeError('timing events were not prequeued; increase --prequeue-cycles')
                end.synchronize(); wall_us=(time.perf_counter_ns()-wall_start)/1000
                for li in range(2):
                    bits=torch.arange(32,device='cuda',dtype=torch.int64)
                    expected_bits=(1+((ids[li].view(-1,1)>>bits)&1)).to(torch.bfloat16)
                    if (not torch.equal(outputs[li][:,:32],expected_bits)
                            or not torch.all(outputs[li][:,32]==1+li).item()
                            or not torch.all(outputs[li][:,33:]==1).item()): raise RuntimeError('row output mismatch')
                    st=stats[li].cpu().tolist(); expected=int(torch.unique(ids[li]).numel()) if n>=4096 else n
                    if st[0] or st[4]!=expected: raise RuntimeError(f'completion mismatch {st} expected {expected}')
                    active=16 if n>=4096 else (4 if a.dispatch=='optimized' and n>16 else 1)
                    lane=lane_stats[li][:active].cpu().tolist()
                    if sum(int(x[4]) for x in lane)!=expected or any(x[0] for x in lane): raise RuntimeError(f'lane status mismatch {lane}')
                if plan.last_active!=active: raise RuntimeError('dispatch mismatch')
                checks+=1
                if i>=5:
                    timings.append(start.elapsed_time(end)*1000); walls.append(wall_us); submits.append(submit_us)
                    counts.append([int(s.cpu()[4]) for s in stats]); lanes.append([x[:active].cpu().tolist() for x in lane_stats])
            samples[str(n)]=timings; wall_samples[str(n)]=walls; submit_samples[str(n)]=submits; counts_by_size[str(n)]=counts; lane_samples[str(n)]=lanes
        decode_overlap=None
        if a.decode_overlap:
            from candidate_readplan import FirstRailContext
            aux=None; aux_graph=None
            aux_stream=torch.cuda.Stream(); gate_stream=torch.cuda.Stream()
            try:
                aux=FirstRailContext(a.library,0,a.port+40,a.rails[0])
                aux_n=64; aux_payload=64; aux_dump=((aux_payload+aux_n*264+63)//64)*64
                aux_arena=torch.empty(aux_dump+64,dtype=torch.uint8,device='cuda')
                if not (aux_arena.data_ptr()+aux_arena.numel()<=arena.data_ptr() or arena.data_ptr()+arena.numel()<=aux_arena.data_ptr()):
                    raise RuntimeError('decode arena aliases prefetch arena')
                if any(aux.handle==c.handle for c in plan.contexts): raise RuntimeError('decode context aliases prefetch context')
                aux.register([aux_arena],False)
                aux.connect_raw(base64.b64decode(md['native_md']),md['addresses'],md['lengths'],md['owner_gpu'])
                aux_ids=torch.arange(aux_n,dtype=torch.int64,device='cuda')
                aux_stats=torch.zeros(8,dtype=torch.uint64,device='cuda')
                aux_out=torch.empty((aux_n,256),dtype=torch.bfloat16,device='cuda')
                stream.synchronize()  # Publish initialization to the independent decode stream.
                def aux_step():
                    aux.fetch_row264(aux_ids,aux_payload,aux_dump,0,aux_stats,0,a.rows)
                    gather_row264(aux_arena.data_ptr()+aux_payload,aux_ids,aux_out,0,a.rows,query_order=True)
                    aux.record_consumed()
                with torch.cuda.stream(aux_stream):
                    aux_step(); aux_stream.synchronize()
                    aux_graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(aux_graph,stream=aux_stream): aux_step()
                torch.cuda.synchronize()
                replay_checks=[]
                for iteration in range(16):
                    with torch.cuda.stream(stream):
                        for li in range(2):
                            index=torch.arange(n,dtype=torch.int64,device='cuda')
                            if a.input_pattern=='paired': index=torch.div(index,2,rounding_mode='floor')
                            ids[li].copy_((index*104729+li*31+(iteration+51)*1031)%a.rows)
                    with torch.cuda.stream(aux_stream):
                        aux_ids.copy_((torch.arange(aux_n,dtype=torch.int64,device='cuda')*104729+(iteration+51)*193)%a.rows)
                    stream.synchronize(); aux_stream.synchronize()
                    gate=torch.cuda.Event()
                    with torch.cuda.stream(gate_stream):
                        torch.cuda._sleep(max(a.prequeue_cycles,40000000)); gate.record(gate_stream)
                    with torch.cuda.stream(aux_stream): aux_stream.wait_event(gate); aux_graph.replay()
                    stream.wait_event(gate); step()
                    if gate.query(): raise RuntimeError('decode overlap launches were not prequeued')
                    stream.synchronize(); aux_stream.synchronize()
                    bits=torch.arange(32,device='cuda',dtype=torch.int64)
                    for li in range(2):
                        expected_bits=(1+((ids[li].view(-1,1)>>bits)&1)).to(torch.bfloat16)
                        st=stats[li].cpu().tolist(); expected=int(torch.unique(ids[li]).numel()) if n>=4096 else n
                        active=16 if n>=4096 else (4 if a.dispatch=='optimized' and n>16 else 1)
                        lane=lane_stats[li][:active].cpu().tolist()
                        if (not torch.equal(outputs[li][:,:32],expected_bits) or not torch.all(outputs[li][:,32]==1+li).item()
                            or not torch.all(outputs[li][:,33:]==1).item() or st[0] or st[4]!=expected
                            or sum(int(x[4]) for x in lane)!=expected or any(x[0] for x in lane)):
                            raise RuntimeError(f'overlap prefetch mismatch layer={li} status={st}')
                    aux_expected_bits=(1+((aux_ids.view(-1,1)>>bits)&1)).to(torch.bfloat16)
                    if (not torch.equal(aux_out[:,:32],aux_expected_bits) or not torch.all(aux_out[:,32:]==1).item()):
                        raise RuntimeError('overlap decode BF16 mismatch')
                    expected_raw=torch.full((aux_n,264),0x38,dtype=torch.uint8,device='cuda')
                    expected_raw[:,:32]=0x38+(((aux_ids.view(-1,1)>>bits)&1).to(torch.uint8)*8)
                    expected_raw[:,256:]=127
                    raw=aux_arena[aux_payload:aux_payload+aux_n*264].view(aux_n,264)
                    ast=aux_stats.cpu().tolist()
                    if not torch.equal(raw,expected_raw) or ast[0] or ast[4]!=aux_n:
                        raise RuntimeError(f'overlap decode raw/status mismatch {ast}')
                    replay_checks.append({'iteration':iteration,'prefetch_completed':[int(x.cpu()[4]) for x in stats], 'decode_completed':int(ast[4])})
                torch.cuda.synchronize(); aux_graph=None; gc.collect(); aux.close(); aux=None
                decode_overlap={'status':'PASS','replays':16,'prefetch_size':n,'decode_size':aux_n,'independent_context':True,
                    'independent_arena':True,'independent_stream':True,'first_rail_channel':0,'checks':replay_checks,
                    'scope':'prequeued independent streams; no measured overlap guarantee, no outer framework capture'}
            finally:
                if aux is not None:
                    torch.cuda.synchronize(); aux_graph=None; gc.collect(); aux.close()
        port_burst=None
        if a.port_burst:
            # Sequential ioctl snapshots are node-wide and may include unrelated traffic.
            from ethtool_port_stats import stats as nic_stats
            def snapshot():
                start=time.monotonic_ns(); ports={}
                for j in range(8):
                    name=f'reth{j}'; values=nic_stats(name)
                    ports[name]={k:int(values[k]) for k in ('rx_bytes_phy','tx_bytes_phy')}
                return {'start_monotonic_ns':start,'end_monotonic_ns':time.monotonic_ns(),'ports':ports}
            def delta(x,y):
                return {name:{key:y['ports'][name][key]-value for key,value in counters.items()} for name,counters in x['ports'].items()}
            stream.synchronize(); before_idle=snapshot(); time.sleep(.3); before_burst=snapshot()
            for _ in range(a.port_burst): step()
            stream.synchronize(); after_burst=snapshot(); time.sleep(.3); after_idle=snapshot()
            for li in range(2):
                bits=torch.arange(32,device='cuda',dtype=torch.int64)
                expected_bits=(1+((ids[li].view(-1,1)>>bits)&1)).to(torch.bfloat16)
                if (not torch.equal(outputs[li][:,:32],expected_bits)
                        or not torch.all(outputs[li][:,32]==1+li).item()
                        or not torch.all(outputs[li][:,33:]==1).item()): raise RuntimeError('port burst output mismatch')
                st=stats[li].cpu().tolist(); expected=int(torch.unique(ids[li]).numel()) if n>=4096 else n
                active=16 if n>=4096 else (4 if a.dispatch=='optimized' and n>16 else 1)
                lane=lane_stats[li][:active].cpu().tolist()
                if st[0] or st[4]!=expected or sum(int(x[4]) for x in lane)!=expected or any(x[0] for x in lane):
                    raise RuntimeError(f'port burst completion mismatch layer={li} status={st} lanes={lane}')
            port_burst={'repetitions':a.port_burst,'size':n,'input_pattern':a.input_pattern,
                        'scope':'node-wide physical interface counters; includes unrelated traffic; sequential ioctl snapshots',
                        'idle_before':before_idle,'burst_start':before_burst,'burst_end':after_burst,'idle_after':after_idle,
                        'delta_idle_before':delta(before_idle,before_burst),'delta_burst':delta(before_burst,after_burst),
                        'delta_idle_after':delta(after_burst,after_idle),'final_output_and_counts':'PASS'}
        # Separate untimed invalid-ID / owner-range boundary and dtype probe.
        probe={'status':'SKIPPED','reason':'requires max size >= 5 and rows >= 8'}
        if maximum>=5 and a.rows>=8:
            hi=a.rows//2
            probe_ids=torch.tensor([0,hi-1,hi,hi+1,a.rows-1],dtype=torch.int64,device='cuda')
            probe_out=torch.empty((5,256),dtype=torch.bfloat16,device='cuda')
            probe_stats=torch.zeros(8,dtype=torch.uint64,device='cuda')
            dtype_rejected=False
            try:
                plan.fetch(probe_ids.to(torch.int32),5,p_off,0,dump,512,probe_stats,weight_index=0,scale_index=0,row_lo=0,row_hi=hi,row264=True)
            except ValueError:
                dtype_rejected=True
            if not dtype_rejected: raise RuntimeError('int32 IDs unexpectedly accepted')
            plan.fetch(probe_ids,5,p_off,0,dump,512,probe_stats,weight_index=0,scale_index=0,row_lo=0,row_hi=hi,row264=True)
            gather_row264(arena.data_ptr()+p_off,probe_ids,probe_out,0,hi,query_order=True,representatives=None)
            plan.record_consumed(); stream.synchronize()
            bits=torch.arange(32,device='cuda',dtype=torch.int64)
            valid=(probe_ids>=0)&(probe_ids<hi)
            expected_bits=torch.where(valid[:,None],1+((probe_ids[:,None]>>bits)&1),0).to(torch.bfloat16)
            expected_out=torch.cat((expected_bits,valid[:,None].expand(-1,224).to(torch.bfloat16)),dim=1)
            if not torch.equal(probe_out,expected_out):
                actual=probe_out.float().cpu(); expected=expected_out.float().cpu()
                diagnostics=[]
                for row in range(5):
                    bad=(actual[row]!=expected[row]).nonzero().flatten()
                    diagnostics.append({'id':int(probe_ids[row].item()),'mismatch_count':int(bad.numel()),
                        'first_mismatch':[(int(col),float(actual[row,col]),float(expected[row,col])) for col in bad[:8]],
                        'actual_first_8':actual[row,:8].tolist(),'expected_first_8':expected[row,:8].tolist()})
                print('BOUNDARY_DIAGNOSTIC '+json.dumps({'rows':diagnostics,'stats':probe_stats.cpu().tolist()},allow_nan=True),flush=True)
                raise RuntimeError('invalid ID / owner boundary output mismatch')
            status=probe_stats.cpu().tolist()
            if status[0] or status[4]!=2: raise RuntimeError(f'boundary completion mismatch {status}')
            probe={'status':'PASS','dtype_int32_rejected':True,'owner_range_ids':[0,hi-1,hi,hi+1,a.rows-1],
                   'row_lo':0,'row_hi':hi,'completed':int(status[4])}
    torch.cuda.synchronize(); mapped_images=loaded_images(); plan.close(); plan=None
    if conn: send(conn,{'status':'drained'}); ack=recv(conn); assert ack['status']=='closed'; conn.close(); conn=None
    if owner: owner.close(); owner=None
    result={'arm':a.arm,'mode':a.mode,'plan':a.plan,'dispatch':a.dispatch,'library':str(Path(a.library).resolve()),'library_sha256':hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),'source_commit':'7fde8d0ac3073a403fef0ef3be401ff7444c4f42','scope':a.role+' synthetic two-layer row264, communication graph only','rows':a.rows,'sizes':a.sizes,'warmup':5,'samples_per_size':20,'checked_replays':checks,'pipeline_depth':2,'configured_lanes':16,'rails':a.rails,'weights':a.weights,'input_pattern':a.input_pattern,'decode_overlap':decode_overlap,'port_burst':port_burst,'loaded_images':mapped_images,'correctness_probe':probe,'samples_us':samples,'host_wall_us_includes_prequeue_wait':wall_samples,'host_submit_us':submit_samples,'lane_stats':lane_samples,'p50_us':{k:statistics.median(v) for k,v in samples.items()},'p90_us_nearest_rank':{k:sorted(v)[math.ceil(.9*len(v))-1] for k,v in samples.items()},'completion_counts':counts_by_size,'environment':{k:v for k,v in os.environ.items() if k.startswith(('ENGRAM_ABI3_','GPUNETIO_','NIXL_PLUGIN_DIR','CUDA_VISIBLE_DEVICES'))}}
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n'); success=True
finally:
    if plan is not None:
        try: torch.cuda.synchronize(); plan.close()
        except Exception: pass
    if conn is not None: conn.close() # Owner retains resources after abnormal termination.
    if owner is not None and success: owner.close()
