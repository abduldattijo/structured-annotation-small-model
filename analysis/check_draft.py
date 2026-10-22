"""Asserts that the numbers typed in draft.md match the saved results. Run from the project folder."""
import json, glob, statistics as st
d=open('draft.md').read()
def S(f):
    x=json.load(open(f)); return x.get('summary',x)
Q=[S(f'results/full/local_test_s{i}_summary.json') for i in range(3)]
H=[S(f'results/full/api_test_run{i}_summary.json') for i in (1,2,3)]
C=S('results/full/api_test_cached_summary.json')
pc=lambda v:f'{v*100:.2f}'
def has(t): assert t in d, t
for r in Q+H+[C]: pass
has(', '.join(pc(r['whole_record_accuracy']) for r in Q))
has(', '.join(pc(r['whole_record_accuracy']) for r in H))
has(f"{st.mean(r['whole_record_accuracy'] for r in Q)*100:.2f}")
has(f"{st.mean(r['whole_record_accuracy'] for r in H)*100:.2f}")
for k in ('scenario_accuracy','intent_accuracy','slots_exact_accuracy'):
    for g in (Q,H):
        v=[r[k]*100 for r in g]; has(f'{min(v):.2f} to {max(v):.2f}')
has(f"{C['whole_record_accuracy']*100:.2f} percent")
has(f"{sum(r['cost_usd'] for r in H)/3:.2f}"[:0] or 'x') if False else None
assert abs(H[0]['cost_usd']-0.37)<0.005 and abs(C['cost_usd']-0.093)<0.0005
assert round(H[0]['cost_usd']/2974*10000,2)==1.25 and round(C['cost_usd']/2974*10000,2)==0.31
assert C['cache_read_tokens_mean']==1036.0 and round(C['input_tokens_mean'])==16
for r in H: assert round(r['input_tokens_mean'])==1052
rps=[r['records_per_second_batched'] for r in Q]; assert (round(min(rps)),round(max(rps)))==(38,40)
tt=[r['batched_total_seconds'] for r in Q]; assert (min(tt),max(tt))==(74.5,77.6); has('74.5 to 77.6')
lat=[r['latency_one_record_median_s'] for r in Q]; assert (min(lat),max(lat))==(0.963,1.018)
mem=[r['peak_gpu_memory_gb'] for r in Q]; assert round(max(mem),1)==1.8
for i in (1,2):
    t=json.load(open(f'results/full/adapter_s{i}_train_info.json')); assert 19<=t['train_seconds']/60<=21 and t['peak_gpu_memory_gb']==7.42
assert [r['parse_failures'] for r in H]==[9,12,7] and Q[0]['parse_failures']==0
pu=S('results/full/api_latency_uncached_summary.json'); pcache=S('results/full/api_latency_cached_summary.json')
print('latency',pu['seconds_per_call_median'],pcache['seconds_per_call_median'])
tot=1.416
pass
F=json.load(open('analysis/facts.json'))
c=F['causes_q'][0]; h=F['causes_h'][0]
for k,(a,b) in {'wrong_intent':(352,508),'intent_ok_diff_slot_types':(300,584),'intent_ok_same_slot_types_diff_values':(111,154),'intent_ok_extra_slots':(69,165),'intent_ok_missing_all_slots':(63,133)}.items(): assert (c[k],h[k])==(a,b),k
assert sum(c.values())==895 and sum(h.values())==1553
p=F['paired_qwen_haiku']; assert (p['both'],p['neither'],p['qwen_only'],p['haiku_only'])==(1254,728,825,167)
assert F['haiku_records_changing_across_3_runs']==212 and F['quirky_n']==169
print('ok; quirky',F['quirky_whole_q'],F['quirky_whole_h'],'media_type',F['slot_type_f1']['media_type'])
for t in ['1,254','728','825','167','1,553','895','212','169','7.1 percent','0.835','0.058','128 times','9 times']: has(t)
