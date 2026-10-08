"""Clean saved-prediction reuse: prespecified average, decisions and horizon curves."""
from study import *
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OLD=ROOT.parent/'JES_review_extensions'

def ci(v):
    v=np.asarray(v,float);return np.percentile(np.random.default_rng(20261006).choice(v,(10000,len(v))).mean(1),[2.5,97.5])
def holm(p):
    p=np.asarray(p,float);idx=np.argsort(p);r=np.empty(len(p));m=0
    for j,i in enumerate(idx):m=max(m,(len(p)-j)*p[i]);r[i]=min(1,m)
    return r
def main():
    cols=['label','origin','cell_id','row','Q_true','Q_pred'];frames=[]
    for origin in ORIGINS:
        p=pd.read_csv(OLD/'results_capacity_only'/f'origin{origin}_pred.csv');frames.append(p[cols])
    for f in ['arima_pred.csv','fair_source_pred.csv']:frames.append(pd.read_csv(OLD/'results'/f)[cols])
    if (ROOT/'results/chronos_pred.csv').exists():frames.append(pd.read_csv(ROOT/'results/chronos_pred.csv')[cols])
    raw=pd.concat(frames,ignore_index=True)
    assert np.isfinite(raw[['Q_true','Q_pred']]).all().all()
    pred=raw.groupby(['label','origin','cell_id','row'])[['Q_true','Q_pred']].mean().reset_index()
    members=['ARIMA_AICc','xgb_tuned_Q_only','Manual_Q_only']
    keys=['origin','cell_id','row'];wide=pred[pred.label.isin(members)].pivot(index=keys,columns='label',values='Q_pred')
    assert list(sorted(wide.columns))==sorted(members) and not wide.isna().any().any()
    avg=wide.mean(axis=1).rename('Q_pred').reset_index();truth=pred[pred.label.eq(members[0])][keys+['Q_true']]
    avg=avg.merge(truth,on=keys,validate='one_to_one');avg['label']='equal_ARIMA_XGB_largeLSTM';pred=pd.concat([pred,avg],ignore_index=True)
    pred['step']=pred.row-pred.origin+1;pred['Q_APE']=100*abs(pred.Q_true-pred.Q_pred)/pred.Q_true
    pred.to_csv(ROOT/'results/saved_clean_predictions_and_average.csv',index=False)
    cell=pred.groupby(['label','origin','cell_id']).Q_APE.mean().reset_index();summ=[]
    for (label,origin),g in cell.groupby(['label','origin']):
        lo,hi=ci(g.Q_APE);summ.append(dict(label=label,origin=origin,n_cells=len(g),Q_MAPE=g.Q_APE.mean(),ci95_low=lo,ci95_high=hi))
    pd.DataFrame(summ).to_csv(ROOT/'tables/saved_clean_combination_summary.csv',index=False)
    comparisons=[]
    for o,g in cell.groupby('origin'):
        w=g.pivot(index='cell_id',columns='label',values='Q_APE')
        for member in members:
            delta=w.equal_ARIMA_XGB_largeLSTM-w[member];lo,hi=ci(delta)
            comparisons.append(dict(origin=o,comparison='equal_average - '+member,n_cells=len(delta),difference_pp=delta.mean(),ci95_low=lo,ci95_high=hi,p_raw=wilcoxon(delta).pvalue))
    c=pd.DataFrame(comparisons);c['p_holm_all9']=holm(c.p_raw);c.to_csv(ROOT/'tables/saved_clean_combination_pairs.csv',index=False)
    d=data();q0=d.groupby('cell_id').Q0.first();prefix={(cid,o):ordered(g).Q.iloc[o-1]/q0[cid] for cid,g in d.groupby('cell_id') if cid in cohort(d) for o in ORIGINS}
    decisions=[]
    for (label,origin,cid),g in pred.groupby(['label','origin','cell_id']):
        g=g.sort_values('row');t=g.Q_true.to_numpy()/q0[cid];p=g.Q_pred.to_numpy()/q0[cid]
        tc=np.flatnonzero(t<.8);pc=np.flatnonzero(p<.8);eligible=prefix[cid,origin]>=.8
        bias=100*(g.Q_pred-g.Q_true)/g.Q_true
        decisions.append(dict(label=label,origin=origin,cell_id=cid,eligible=eligible,event_true=bool(len(tc)),event_pred=bool(len(pc)),last_APE=g.Q_APE.iloc[-1],mean_signed_bias_pct=bias.mean(),last_signed_bias_pct=bias.iloc[-1],optimistic_row_share=(g.Q_pred>g.Q_true).mean(),true_cross_step=int(tc[0]+1) if len(tc) else len(g)+1,pred_cross_step=int(pc[0]+1) if len(pc) else len(g)+1,both_cross_step_abs_error=abs(int(tc[0])-int(pc[0])) if len(tc) and len(pc) and eligible else np.nan))
    D=pd.DataFrame(decisions);D.to_csv(ROOT/'tables/saved_clean_decision_cells.csv',index=False);out=[]
    for (lab,o),g in D.groupby(['label','origin']):
        e=g[g.eligible];tp=int((e.event_true&e.event_pred).sum());fp=int((~e.event_true&e.event_pred).sum());fn=int((e.event_true&~e.event_pred).sum());tn=int((~e.event_true&~e.event_pred).sum())
        out.append(dict(label=lab,origin=o,n_cells=len(g),eligible_cells=len(e),already_below80_cells=len(g)-len(e),true_events=tp+fn,tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else np.nan,recall=tp/(tp+fn) if tp+fn else np.nan,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,last_APE=g.last_APE.mean(),mean_signed_bias_pct=g.mean_signed_bias_pct.mean(),optimistic_row_share=g.optimistic_row_share.mean(),both_cross_step_abs_error=e.both_cross_step_abs_error.mean(),censored_cross_abs_error=abs(e.pred_cross_step-e.true_cross_step).mean()))
    pd.DataFrame(out).to_csv(ROOT/'tables/saved_clean_decision_summary.csv',index=False)
    rows=[]
    for (lab,o,s),g in pred.groupby(['label','origin','step']):
        lo,hi=ci(g.Q_APE);rows.append(dict(label=lab,origin=o,step=s,n_cells=g.cell_id.nunique(),Q_APE_mean=g.Q_APE.mean(),ci95_low=lo,ci95_high=hi))
    H=pd.DataFrame(rows);H.to_csv(ROOT/'tables/saved_clean_error_by_horizon.csv',index=False)
    labels=['ARIMA_AICc','xgb_tuned_Q_only','rf_tuned_Q_only','Manual_Q_only','CPUwin_NSGA-II_Q_only','gru_tuned_Q_only','sqrt_days_lag1','persistence','equal_ARIMA_XGB_largeLSTM']
    names=['ARIMA','XGBoost','RF','Large LSTM','Compact LSTM','GRU','Lagged sqrt-time','Persistence','Equal average']
    if (ROOT/'results/chronos_pred.csv').exists():labels.append('ChronosBolt_small_zero');names.append('Chronos-Bolt small')
    fig,axes=plt.subplots(1,3,figsize=(16,5),sharey=True)
    for ax,o in zip(axes,ORIGINS):
        for lab,name in zip(labels,names):
            h=H[H.origin.eq(o)&H.label.eq(lab)].sort_values('step');ax.plot(h.step,h.Q_APE_mean,label=name,linewidth=2 if lab.startswith('equal_') else 1.2)
        ax.set_title(f'{o} measured checkups');ax.set_xlabel('Checkups ahead');ax.grid(alpha=.25)
    axes[0].set_ylabel('Mean capacity APE (%)');axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=5,fontsize=9)
    fig.suptitle('Capacity-only frozen-configuration sensitivity and prespecified average\nAvailable-cell means at each horizon; seed predictions averaged before scoring',fontsize=12)
    fig.tight_layout(rect=[0,.10,1,.90]);fig.savefig(ROOT/'tables/saved_clean_error_by_horizon.png',dpi=200,bbox_inches='tight');fig.savefig(ROOT/'tables/saved_clean_error_by_horizon.pdf',bbox_inches='tight');plt.close(fig)
    manifest={str(p.relative_to(OLD)):sha(p) for p in [*[OLD/'results_capacity_only'/f'origin{o}_pred.csv' for o in ORIGINS],OLD/'results/arima_pred.csv',OLD/'results/fair_source_pred.csv']}
    receipt=dict(passed=all(pred[pred.origin.eq(o)].groupby('label').cell_id.nunique().eq(104).all() for o in ORIGINS),origin_list=ORIGINS,models=len(pred.label.unique()),combination_members=members,weights=[1/3]*3,tuning='None; members/weights prespecified before analysis',seed_aggregation='Average prediction across three training seeds, then score; differs from averaging seed-wise errors.',decision_eligibility='Last measured checkup above/equal80% Q0, so first future-step crossings count.',interval_status='Not calculated from these outer-test residuals; requires independent calibration outputs.',source_hashes=manifest)
    (ROOT/'audit/saved_analysis_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');assert receipt['passed'];print('SAVED_CLEAN_ANALYSIS_PASS',flush=True)
if __name__=='__main__':main()
