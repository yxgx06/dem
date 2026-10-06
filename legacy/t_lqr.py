import sys;sys.path.insert(0,'.')
import lib3 as L, numpy as np
def r(c,**k):
    ey=L.run(c,**k);bad=(not np.all(np.isfinite(ey))) or np.max(abs(ey))>1.5
    return 'DIV' if bad else f'{np.max(abs(ey))*100:.2f}'
print('Vx',L.Vx)
for d in [14,16,20,25]: print('delay',d*10,{c:r(c,delay=d) for c in ['pid','mpc','lqr']})
print('mass1.3',{c:r(c,mass_scale=1.3) for c in ['pid','mpc','rl','lqr']})
print('mass1.9',{c:r(c,mass_scale=1.9) for c in ['pid','mpc','rl','lqr']})
print('noise2cm',{c:r(c,noise=0.02) for c in ['pid','mpc','rl','lqr']})
print('noise5mm',{c:r(c,noise=0.005) for c in ['pid','mpc','rl','lqr']})
print('nominal',{c:r(c) for c in ['pid','mpc','rl','lqr']})
