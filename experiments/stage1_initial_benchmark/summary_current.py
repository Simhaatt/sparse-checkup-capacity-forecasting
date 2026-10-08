"""Verified completed baselines/population rows for the required summary CSV."""
from r3 import *
from scipy.stats import wilcoxon

def ci(values):
    x=np.asarray(values,float);return np.percentile(np.random.default_rng(20261005).choice(x,(10000,len(x))).mean(1),[2.5,97.5])
def holm(p):
    p=np.asarray(p,float);idx=np.argsort(p);adjusted=np.empty(len(p));high=0.
    for k,i in enumerate(idx):high=max(high,(len(p)-k)*p[i]);adjusted[i]=min(1.,high)
    return adjusted
def main():
    metadata('current_summary');frames=[]
    for f in ['reused_arima_pred.csv','reused_fair_source_pred.csv','reused_chronos_pred.csv']:
        p=pd.read_csv(ROOT/'results'/f);p['seed']=0;p['experiment']='capacity_baselines';p=p.rename(columns={'label':'method'});frames.append(p[['method','origin','seed','cell_id','row','Q_true','Q_pred','experiment']])
    path=ROOT/'results/semiempirical_pred_v3.csv'
    if path.exists() and (ROOT/'audit/semiempirical_receipt_v3.json').exists():
        p=pd.read_csv(path);p=p[p.role.eq('test')].copy();p['experiment']='population_semiempirical';frames.append(p[['method','origin','seed','cell_id','row','Q_true','Q_pred','experiment']])
    p=pd.concat(frames,ignore_index=True);assert not p.duplicated(['method','origin','seed','cell_id','row']).any()
    p['APE']=100*abs(p.Q_pred-p.Q_true)/p.Q_true
    cell=p.groupby(['method','origin','experiment','seed','cell_id']).APE.mean().groupby(['method','origin','experiment','cell_id']).mean().reset_index()
    assert cell.groupby(['method','origin','experiment']).cell_id.nunique().eq(104).all()
    rows=[]
    for (method,origin,experiment),g in cell.groupby(['method','origin','experiment']):
        lo,hi=ci(g.APE);rows.append(dict(method=method,origin=origin,experiment=experiment,status='done',n_cells=len(g),Q_MAPE_equal_cell=g.APE.mean(),ci95_low=lo,ci95_high=hi,cell_sd=g.APE.std(),aggregation='Stepmeanperseed/cell, seedmeanwithincell, equalcells',interpretation='Completedbaseline; primarytunedcomparison pending'))
    pd.DataFrame(rows).to_csv(ROOT/'RESULTS_SUMMARY.csv',index=False);cell.to_csv(ROOT/'tables/completed_baseline_cells.csv',index=False)
    tests=[]
    for origin,g in cell.groupby('origin'):
        pivot=g.pivot(index='cell_id',columns='method',values='APE');ref='ARIMA_AICc';family=f'completed_capacity_baselines_vs_ARIMA_origin{origin}'
        for method in sorted(set(pivot.columns)-{ref}):
            delta=pivot[method]-pivot[ref];lo,hi=ci(delta);tests.append(dict(family=family,origin=origin,comparison=method+' - '+ref,n_cells=len(delta),difference_pp=delta.mean(),ci95_low=lo,ci95_high=hi,p_raw=wilcoxon(delta).pvalue if np.any(delta) else 1.))
    t=pd.DataFrame(tests)
    for family,g in t.groupby('family'):t.loc[g.index,'p_Holm']=holm(g.p_raw);t.loc[g.index,'family_size']=len(g)
    t.to_csv(ROOT/'tables/completed_baseline_pairs.csv',index=False)
    (ROOT/'audit/current_summary_receipt.json').write_text(json.dumps(dict(passed=True,summary_rows=len(rows),cells_per_method_origin=104,bootstrap_draws=10000,bootstrap_seed=20261005,aggregation='Seedwiseerrorsfirst, never point-averageacrossseeds for primarymetric',scope='Completedcapacitybaselines andcorrectedpopulationlaw only; notproofthatmainretuning orotherround3requirementsarecomplete'),indent=2),encoding='utf8');print('CURRENT_SUMMARY_VERIFIED',len(rows),flush=True)
if __name__=='__main__':main()
