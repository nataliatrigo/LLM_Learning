"""Certify an exact rational partition of the broad calibration box near rho=.7.

Run: uv run --with python-flint python -B forgetting/Code/numerical_validations/local_threshold_intervals/cover070.py
Every leaf is either independently certified or explicitly unresolved; no
sampling or inference from adjacent boxes is used. Shared-demand cache entries
are mathematically identical inputs to the same validated oracle.
"""
from __future__ import annotations
import argparse
from collections import deque
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import time
import certify_box as verifier

OUT=Path(__file__).resolve().parents[4] / 'forgetting/outputs/numerical_validations/local_threshold_intervals'
KEYS=('rho','p0','gamma','p1','p2','cost_ratio')


def bisect(box,key):
    lo,hi=map(Fraction,box[key]);mid=(lo+hi)/2
    result=[]
    for a,b in [(lo,mid),(mid,hi)]:
        child={k:list(v) for k,v in box.items()}
        child[key]=[str(a),str(b)]
        result.append(child)
    return result


def volume(box):
    v=Fraction(1)
    for key in KEYS:
        lo,hi=map(Fraction,box[key]);v*=hi-lo
    return v


def verify_partition(original,leaves):
    assert sum((volume(b) for b in leaves),Fraction(0))==volume(original)
    for box in leaves:
        for key in KEYS:
            ol,oh=map(Fraction,original[key]);lo,hi=map(Fraction,box[key])
            assert ol<=lo<hi<=oh
    for i,a in enumerate(leaves):
        for b in leaves[i+1:]:
            assert any(max(Fraction(a[k][0]),Fraction(b[k][0]))>=
                       min(Fraction(a[k][1]),Fraction(b[k][1])) for k in KEYS)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--nodes',type=int,default=4096)
    parser.add_argument('--max-leaves',type=int,default=64)
    parser.add_argument('--out',default=str(OUT/'cover070_results.json'))
    args=parser.parse_args()
    original=json.loads((OUT/'box070.json').read_text())
    start=time.monotonic()
    oracle=verifier.demand_bounds
    cache={}
    def cached_oracle(n,rho,p0):
        key=(n,tuple(map(str,rho)),tuple(map(str,p0)))
        if key not in cache:
            value=oracle(n,rho,p0)
            for label in ['nodes','cells','derivative']:
                for arr in value[label]:arr.setflags(write=False)
            cache[key]=value
        return cache[key]
    verifier.demand_bounds=cached_oracle
    first=[]
    for r,rb in enumerate(bisect(original,'rho')):
        for p,pb in enumerate(bisect(rb,'p0')):
            first.append((f'r{r}p{p}',pb,0))
    queue=deque(first);certified=[];unresolved=[];attempts=[];splits=[]
    max_leaves=args.max_leaves
    # Initial rho,p0 bisections are followed by gamma,p1,p2, then another
    # rho,p0 pass if needed. All midpoint splits use exact rational arithmetic.
    split_order=('gamma','p1','p2','rho','p0','cost_ratio')
    def save():
        leaves=[v['box'] for v in certified+unresolved]+[b for _,b,_ in queue]
        verify_partition(original,leaves)
        payload=dict(original_box=original,nodes=args.nodes,maximum_leaves=max_leaves,
                     complete=not queue and not unresolved,
                     certified_boxes=len(certified),unresolved_boxes=len(unresolved),
                     pending_boxes=len(queue),attempt_count=len(attempts),
                     demand_cache_entries=len(cache),elapsed_seconds=time.monotonic()-start,
                     exact_partition_verified=True,
                     method='Independent outward-rounded Bellman value and derivative enclosure on every rational leaf box',
                     verifier_sha256=hashlib.sha256(Path(__file__).with_name('certify_box.py').read_bytes()).hexdigest(),
                     oracle_sha256=hashlib.sha256(Path(__file__).with_name('demand_bounds.py').read_bytes()).hexdigest(),
                     certified=certified,unresolved=unresolved,
                     pending=[dict(path=p,box=b,depth=d) for p,b,d in queue],
                     splits=splits,attempts=attempts)
        if certified:
            payload['weakest_derivative_upper']=max(v.get('derivative_upper',-float('inf')) for v in certified)
            payload['minimum_advantage_zero_lower']=min(v.get('advantage_at_zero_lower',float('inf')) for v in certified)
            payload['maximum_advantage_one_upper']=max(v.get('advantage_at_one_upper',-float('inf')) for v in certified)
            brackets=[v['bracket'] for v in certified if 'bracket' in v]
            if brackets:payload['uniform_threshold_bracket']=[min(b[0] for b in brackets),max(b[1] for b in brackets)]
        Path(args.out).write_text(json.dumps(payload,indent=2)+'\n')
    while queue:
        path,box,depth=queue.popleft()
        result=verifier.certify(box,N=args.nodes)
        result.update(path=path,split_depth=depth)
        attempts.append(result)
        print(json.dumps(dict(path=path,certified=result['certified'],
                              derivative_upper=result.get('derivative_upper'),
                              max_value_width=result['max_value_width'],
                              seconds=result['seconds'])),flush=True)
        if result['certified']:
            certified.append(result)
        elif len(certified)+len(unresolved)+len(queue)+2<=max_leaves:
            key=split_order[depth%len(split_order)]
            children=bisect(box,key)
            splits.append(dict(path=path,key=key,midpoint=children[0][key][1]))
            for i,child in enumerate(children):queue.append((path+f'/{key}{i}',child,depth+1))
        else:
            unresolved.append(result)
        save()
    print(json.dumps(dict(complete=not unresolved,certified=len(certified),
                          unresolved=len(unresolved),attempts=len(attempts),
                          elapsed_seconds=time.monotonic()-start,
                          result_path=args.out)),flush=True)


if __name__=='__main__':main()
