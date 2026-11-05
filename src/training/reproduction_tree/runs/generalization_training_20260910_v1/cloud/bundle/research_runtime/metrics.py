import numpy as np

def scores(y, users, logits):
    y, users, logits = np.asarray(y), np.asarray(users), np.asarray(logits)
    if logits.shape != (len(y),40) or not np.isfinite(logits).all(): raise ValueError('INVALID_LOGITS')
    pred = logits.argmax(1); cm = np.zeros((40,40), np.int64); np.add.at(cm,(y,pred),1)
    den = cm.sum(0)+cm.sum(1); f1 = np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den>0)
    z = logits.astype(np.float64); z -= z.max(1,keepdims=True); prob = np.exp(z); prob /= prob.sum(1,keepdims=True)
    per = {str(u):dict(n=int((users==u).sum()),correct=int((pred[users==u]==y[users==u]).sum())) for u in sorted(set(users.tolist()))}
    return dict(n=len(y),correct=int((pred==y).sum()),accuracy=float((pred==y).mean()),macro_f1=float(f1.mean()),
                user_equal=float(np.mean([v['correct']/v['n'] for v in per.values()])),per_user=per,
                nll=float(-np.log(prob[np.arange(len(y)),y].clip(1e-300)).mean()),
                brier=float(np.square(prob-np.eye(40)[y]).sum(1).mean()))

def delta(candidate, control):
    per = {u:candidate['per_user'][u]['correct']/v['n']-v['correct']/v['n'] for u,v in control['per_user'].items()}
    return dict(accuracy=candidate['accuracy']-control['accuracy'],macro_f1=candidate['macro_f1']-control['macro_f1'],
                user_equal=candidate['user_equal']-control['user_equal'],per_user=per)

def adjudicate_ce(records):
    # records: fixed two seeds, each pooling the two preregistered development folds.
    if len(records) != 2: return dict(status='INCOMPLETE',scientific_negative=False)
    vs_control = [delta(r['candidate'],r['control']) for r in records]
    vs_base = [delta(r['candidate'],r['baseline']) for r in records]
    avg = lambda values: float(np.mean(values))
    user_names = sorted(vs_control[0]['per_user'])
    mean_user = {u:avg([v['per_user'][u] for v in vs_control]) for u in user_names}
    leave_best = (sum(mean_user.values())-max(mean_user.values()))/(len(mean_user)-1)
    checks = dict(mean_gain_vs_baseline=avg([d['accuracy'] for d in vs_base])>=.01-1e-12,
                  mean_gain_vs_control=avg([d['accuracy'] for d in vs_control])>=.005-1e-12,
                  each_seed_positive=all(d['accuracy']>0 for d in vs_control),
                  each_fold_mean_nonnegative=all(avg([r['fold_delta'][f] for r in records])>=-1e-12 for f in ['0','3']),
                  macro_nonnegative=avg([d['macro_f1'] for d in vs_control])>=-1e-12,
                  user_equal_nonnegative=avg([d['user_equal'] for d in vs_control])>=-1e-12,
                  worst_user_mean=min(mean_user.values())>=-.03-1e-12,
                  no_single_user_catastrophe=min(v for d in vs_control for v in d['per_user'].values())>=-.10-1e-12,
                  leave_best_user_out=leave_best>=-1e-12)
    if all(checks.values()): status='CONFIRMED_REUSED_DEVELOPMENT_NO_DEPLOYMENT'
    elif all(d['accuracy']<=0 for d in vs_control) or not checks['no_single_user_catastrophe']: status='NEGATIVE_EXACT_CONFIG_CLOSED'
    else: status='INCONCLUSIVE_NO_AUTOMATIC_RETRY'
    return dict(status=status,checks=checks,vs_control=vs_control,vs_baseline=vs_base,mean_user_delta=mean_user,
                leave_best_user_out=leave_best,positive_users=sum(v>0 for v in mean_user.values()),
                independent_unseen_user_evidence=False,automatic_promotion=False)
