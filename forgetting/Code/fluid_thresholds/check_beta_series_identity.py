import math,json
from scipy.integrate import quad
from scipy.special import digamma, betaln, betaincc
from pathlib import Path
cases=[(1.,1.,.5),(1.3,3.7,.31),(10.,12.,.5),(1.,21.,.02),(21.,1.,.9),(101.,21.,.8)]
rows=[]
for a,b,p in cases:
 q=1-p;mu=digamma(a)-digamma(b)
 def score(t):
  if t==0 or t==1:return 0.
  return math.exp((a-1)*math.log(t)+(b-1)*math.log1p(-t)-betaln(a,b))*(math.log(t)-math.log1p(-t)-mu)
 direct=quad(score,p,1,epsabs=1e-13,epsrel=3e-12,limit=300)[0] if p>a/(a+b) else -quad(score,0,p,epsabs=1e-13,epsrel=3e-12,limit=300)[0]
 def outer(z):
  def inner(s):
   return (1-s)**(a-1)*(1-s*z)**(b-1)/(p+q*z)+(1-s*z)**(a-1)*(1-s)**(b-1)/(p*z+q)
  return quad(inner,0,1,epsabs=1e-12,epsrel=3e-12,limit=200)[0]
 integ=quad(outer,0,1,epsabs=1e-12,epsrel=3e-12,limit=200)[0]
 identity=math.exp(a*math.log(p)+b*math.log1p(-p)-betaln(a,b))*integ
 h=1e-4
 finite=(betaincc(a-2*h,b+2*h,p)-8*betaincc(a-h,b+h,p)+8*betaincc(a+h,b-h,p)-betaincc(a+2*h,b-2*h,p))/(12*h)
 row=dict(a=a,b=b,p=p,score_integral=direct,series_via_rectangle_integral=identity,five_point_parameter_difference=finite,identity_relative_error=abs(identity-direct)/direct)
 rows.append(row);print(json.dumps(row),flush=True)
(Path(__file__).resolve().parents[3] / 'forgetting/outputs/fluid_thresholds' / 'beta_series_identity_check_20260927.json').write_text(json.dumps(rows,indent=2))
