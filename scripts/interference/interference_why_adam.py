"""Why does MAttr+Adam degrade with more steps on the interference task? ANSWERED: diffusion.

    uv run python scripts/interference/interference_why_adam.py     # ~35 min, CPU; needs --tag hard on disk

Two hypotheses, one discriminator (the lr sweep). The answer is H-noise; see
docs/interference_toy.md for the numbers and the reading.

H-noise : Adam normalises by sqrt(v), so a unit whose gradient is pure noise gets the SAME
          step size as a unit carrying signal. Over T steps the ~16k interference units random
          -walk with step ~lr, so their scores spread as ~lr*sqrt(T) and start overtaking the
          weakly-real units in the tail. Predicts: interference-score spread GROWS with T and
          with lr; SGD (steps scaled by real gradient magnitude) does not do this.
H-objective : the mask objective E_k[L(U*m)] under a log-uniform k just is not dL, and better
          minimisation buys a ranking tuned to small k at the tail's expense. Predicts: the
          decline is lr-INDEPENDENT once converged, and the loss keeps falling as rho falls.

Discriminator: sweep Adam's lr. H-noise says the decline tracks lr; H-objective says it does not.
"""
import sys, json, torch
sys.path.insert(0,'scripts/interference')
import interference_attrib as IA
import interference_toy as IT
from interference_toy import sample_x, sweep
from learning_to_attribute import learn_scores

TAG, SNAPS, BATCH = 'hard', (256,1000,3000,10000), 2048
d = IT.OUT.parent / f'{IT.OUT.name}_{TAG}'
m = torch.load(d/'model.pt'); U,b,A,v,dl = m['U'],m['b'],m['A'],m['v'],m['dl']
n = U.shape[0]; real = (dl>1e-4).reshape(-1)

def run(opt, lr, seed=0):
    g = torch.Generator().manual_seed(seed+40_000); out={}; losses=[]
    def loss_fn(mask): return IA.loss_on(U*mask.view(n,n), b, A, v, sample_x(BATCH,g))
    def on_step(step,k,loss,scores):
        losses.append(loss)
        if (step+1) in SNAPS: out[step+1]=scores.detach().clone()
    learn_scores(n*n, loss_fn, steps=max(SNAPS), variant='topk', k_schedule='log',
                 T=0.5, lr=lr, optimizer=opt, on_step=on_step, log_every=1)
    return out, losses

for opt, lr in (('adam',0.05),('adam',0.01),('adam',0.002),('sgd',1.0)):
    traj, losses = run(opt, lr)
    print(f'=== {opt} lr={lr} ===', flush=True)
    for T_ in SNAPS:
        s = traj[T_]
        c = sweep(s.view(n,n), dl); rec,p = c['recall'], c['precision']
        p80 = p[next(i for i,x in enumerate(rec) if x>=.8)]
        ra=s.argsort().argsort().float(); rb=dl.reshape(-1).argsort().argsort().float()
        rho=float(((ra-ra.mean())/ra.std()*(rb-rb.mean())/rb.std()).mean())
        sat = float((s.abs()/0.5 > 4).float().mean())     # gate slope < 0.018
        lo = sum(losses[T_-100:T_])/100
        print(f'  T={T_:>6} rho {rho:+.3f} P@R.8 {p80:.3f} | interf-score std {s[~real].std():.4f} '
              f'real-score mean {s[real].mean():+.4f} interf mean {s[~real].mean():+.4f} '
              f'| |s|max {s.abs().max():.2f} sat {sat:.3f} | loss {lo:.4f}', flush=True)
