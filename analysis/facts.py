import json, sys, statistics as st
from collections import Counter
sys.path.insert(0,'colab_scripts')
from schema import parse, score_record
def load(f):
    return [json.loads(l) for l in open(f)]
def scored(rows):
    out={}
    for r in rows:
        p,clean=parse(r['raw'])
        out[r['id']]=(r,p,score_record(r['gold'],p),clean)
    return out
Q=[scored(load(f'results/full/local_test_s{i}.jsonl')) for i in range(3)]
H=[scored(load(f'results/full/api_test_run{i}.jsonl')) for i in (1,2,3)]
C=scored(load('results/full/api_test_cached.jsonl'))
ids=list(Q[0])
def whole(d): return {i:d[i][2]['whole'] for i in d}
F={}
# paired, seed0 vs run1
for name,q,h in [('s0_run1',Q[0],H[0])]:
    c=Counter((q[i][2]['whole'],h[i][2]['whole']) for i in ids)
    F['paired_qwen_haiku']={'both':c[(1,1)],'qwen_only':c[(1,0)],'haiku_only':c[(0,1)],'neither':c[(0,0)]}
# ever-right across all 3 haiku / qwen
F['n']=len(ids)
# error causes among wrong records
def causes(d):
    c=Counter()
    for i in ids:
        r,p,s,_=d[i]
        if s['whole']: continue
        if p is None: c['unparseable']+=1
        elif not s['intent'] or not s['scenario']: c['wrong_intent']+=1
        else:
            g=r['gold']['slots']; pr=p['slots']
            if not g and pr: c['intent_ok_extra_slots']+=1
            elif g and not pr: c['intent_ok_missing_all_slots']+=1
            else:
                gt=Counter(t for t,_ in g); pt=Counter(t for t,_ in pr)
                if gt==pt: c['intent_ok_same_slot_types_diff_values']+=1
                else: c['intent_ok_diff_slot_types']+=1
    return dict(c)
F['causes_q']=[causes(q) for q in Q]
F['causes_h']=[causes(h) for h in H]
# intent wrong but scenario right share
def scen_ok_intent_wrong(d):
    return sum(1 for i in ids if d[i][2]['scenario'] and not d[i][2]['intent'])
F['scen_ok_intent_wrong_q']=[scen_ok_intent_wrong(q) for q in Q]
F['scen_ok_intent_wrong_h']=[scen_ok_intent_wrong(h) for h in H]
# invalid labels (outside inventory)
inv_i=set(); inv_s=set()
tr=[json.loads(l) for l in open('results/full/api_test_run1_summary.json')] if False else None
# gold records with no slots
nos=[i for i in ids if not Q[0][i][0]['gold']['slots']]
F['gold_noslot_n']=len(nos)
def noslot_acc(d): return sum(d[i][2]['slots_exact'] for i in nos)/len(nos)
F['noslot_slots_exact_q']=[round(noslot_acc(q),4) for q in Q]
F['noslot_slots_exact_h']=[round(noslot_acc(h),4) for h in H]
sl=[i for i in ids if Q[0][i][0]['gold']['slots']]
def slot_acc(d): return sum(d[i][2]['slots_exact'] for i in sl)/len(sl)
F['withslot_n']=len(sl)
F['withslot_slots_exact_q']=[round(slot_acc(q),4) for q in Q]
F['withslot_slots_exact_h']=[round(slot_acc(h),4) for h in H]
# per scenario whole acc (mean over runs)
scs=sorted({Q[0][i][0]['gold']['scenario'] for i in ids})
per={}
for s in scs:
    sid=[i for i in ids if Q[0][i][0]['gold']['scenario']==s]
    qa=st.mean(sum(q[i][2]['whole'] for i in sid)/len(sid) for q in Q)
    ha=st.mean(sum(h[i][2]['whole'] for i in sid)/len(sid) for h in H)
    per[s]=(len(sid),round(qa,3),round(ha,3))
F['per_scenario']=per
# per slot type recall-ish: type-level F1
def slot_type_stats(d):
    tp=Counter(); gp=Counter(); pp=Counter()
    for i in ids:
        r,p,s,_=d[i]
        g=Counter(map(tuple,r['gold']['slots'])); pr=Counter(p['slots']) if p else Counter()
        for k,v in g.items(): gp[k[0]]+=v
        for k,v in pr.items(): pp[k[0]]+=v
        for k,v in (g&pr).items(): tp[k[0]]+=v
    return tp,gp,pp
tq=[slot_type_stats(q) for q in Q]; th=[slot_type_stats(h) for h in H]
types=sorted(tq[0][1], key=lambda t:-tq[0][1][t])[:8]
def f1(tp,gp,pp,t):
    p=tp[t]/pp[t] if pp[t] else 0; r=tp[t]/gp[t] if gp[t] else 0
    return round(2*p*r/(p+r),3) if p+r else 0
F['slot_type_f1']={t:(tq[0][1][t], round(st.mean(f1(*x,t) for x in tq),3), round(st.mean(f1(*x,t) for x in th),3)) for t in types}
# hardest intent confusions for each
def conf(d):
    c=Counter()
    for i in ids:
        r,p,s,_=d[i]
        if p and not s['intent']: c[(r['gold']['intent'],p['intent'])]+=1
    return c.most_common(5)
F['conf_q0']=conf(Q[0]); F['conf_h1']=conf(H[0])
# examples
import random
random.seed(7)
def ex(d,filt,k=4):
    xs=[i for i in ids if filt(d[i])]
    return [(d[i][0]['text'],d[i][0]['gold'],d[i][1]) for i in random.sample(xs,min(k,len(xs)))]
F['ex_haiku_unparse']=[(H[0][i][0]['text'],H[0][i][0]['raw'][:100]) for i in ids if H[0][i][1] is None][:5]
F['ex_q_unparse']=sum(1 for i in ids if Q[0][i][1] is None)
# haiku across runs: records whose whole score differs among 3 runs
diff=sum(1 for i in ids if len({h[i][2]['whole'] for h in H})>1)
F['haiku_records_changing_across_3_runs']=diff
# text length
F['mean_words']=round(st.mean(len(Q[0][i][0]['text'].split()) for i in ids),2)
# Haiku wrong intent where gold in general_quirky
F['quirky_n']=sum(1 for i in ids if Q[0][i][0]['gold']['intent']=='general_quirky')
F['quirky_whole_q']=[round(sum(q[i][2]['whole'] for i in ids if q[i][0]['gold']['intent']=='general_quirky')/F['quirky_n'],3) for q in Q]
F['quirky_whole_h']=[round(sum(h[i][2]['whole'] for i in ids if h[i][0]['gold']['intent']=='general_quirky')/F['quirky_n'],3) for h in H]
json.dump(F,open('analysis/facts.json','w'),indent=1,default=str)
print(json.dumps(F,indent=1,default=str))
