from search_agent import solve,valleys,OUT
import numpy as np,json
from scipy.special import betaincc
rho=.6;h=1-rho
for p0 in [.85,.9,.95,.975,.99,.995,.999,.9995,.9999]:
 d=betaincc(1+.3/h,1+.7/h,p0);g=1-d
 for eps in [.2,.1,.05,.025,.01,.005,.001]:
  par=[rho,p0,eps,2*eps,.25,.25+1.125*eps,g]
  x,v,a,it=solve(par,n=8001)
  m,p,va=valleys(a,x)
  result=dict(par=par,margin=m,points=p,advantages=va,index=[p0,eps],n=8001,it=it)
  print(json.dumps(result),flush=True)
  if m>1e-5:(OUT/f'search_agent_limit_{p0}_{eps}.json').write_text(json.dumps(result,indent=2))
