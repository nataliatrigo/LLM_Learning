from hashlib import sha256
from pathlib import Path
import time
import numpy as np
from oblivious.oe_distribution_fixed_p0.code.conditional_model import FixedOutsideConditionalExperiment
from oblivious.oe_distribution_fixed_p0.code.conditional_refine_case import source_fingerprints
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.run_experiment import save_bundle

started=time.monotonic(); sources=source_fingerprints()
n=151; e=FixedOutsideConditionalExperiment(n,Parameters(rho=.97),p0=.5,order=256)
alpha=np.zeros(n);alpha[73:106]=1
print('Pure proposal alpha73..105=1, every other node0; begin fresh full positive/reversed certification',flush=True)
r=e.evaluate(alpha)
regret=np.maximum(r['advantage'],0)-alpha*r['advantage']
bad=np.flatnonzero(regret>1e-10)
print('FULL',r['verified'],r['dp_diagnostics'],flush=True)
print('bad',bad.tolist(),'alpha',alpha[bad].tolist(),'advantage',r['advantage'][bad].tolist(),flush=True)
print('boundary',[(k,float(r['advantage'][k])) for k in (70,71,72,73,74,104,105,106,107,108)],flush=True)
if sources!=source_fingerprints():raise RuntimeError('Numerical source changed')
save_bundle(Path('/tmp/conditional_round_0p97_n151'),dict(
 competition=r,configuration=e.configuration(),elapsed_seconds=time.monotonic()-started,
 source_fingerprints=sources,policy_sha256=sha256(alpha.tobytes()).hexdigest(),
 proposer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
 search_metadata=dict(method='Rounded certified n101 interpolation; fresh unchanged full conditional certificate',
  exhaustive=False,unique=False)))
print('SAVED /tmp/conditional_round_0p97_n151; elapsed',time.monotonic()-started,flush=True)
