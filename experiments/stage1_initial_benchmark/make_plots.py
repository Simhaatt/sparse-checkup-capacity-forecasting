"""Standalone research figures from verified numerical tables."""
from r3 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG=ROOT/'figures'
def save(fig,name,sources):
    FIG.mkdir(exist_ok=True);fig.savefig(FIG/f'{name}.png',dpi=200,bbox_inches='tight');fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    return dict(figure=name,png=f'figures/{name}.png',pdf=f'figures/{name}.pdf',sources=';'.join(sources))
def main():
    metadata('plots');manifest=[];plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    labels=FAMILIES+['ARIMA_AICc','avg_ARIMA_XGB_NN','regime_origin_selector','ChronosBolt_small_zero','semiempirical']
    names=['Compact LSTM','GRU','Large LSTM','XGBoost','RF','ARIMA','Equal mean','Selector','Chronos small','Population law']
    H=pd.read_csv(ROOT/'tables/error_by_horizon.csv');fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    for ax,origin in zip(axes,[10,15,20]):
        for method,name in zip(labels,names):
            h=H[H.method.eq(method)&H.origin.eq(origin)].sort_values('step');ax.plot(h.step,h.Q_APE,label=name,lw=1.4)
        ax.set_title(f'{origin} measured checkups');ax.set_xlabel('Checkups ahead');ax.grid(alpha=.2)
    axes[0].set_ylabel('Equal-cell capacity APE (%)');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5,fontsize=8);fig.tight_layout(rect=[0,.14,1,1]);manifest.append(save(fig,'capacity_error_horizon',['tables/error_by_horizon.csv']))
    C=pd.read_csv(ROOT/'tables/conformal_horizon_overall.csv');fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
    for col,level in enumerate([.8,.9]):
        for method,name in zip(labels,names):
            h=C[C.method.eq(method)&C.origin.eq(20)&np.isclose(C.nominal,level)].sort_values('step');axes[0,col].plot(h.step,h.coverage,label=name)
            finite=h.mean_width_SOH.replace([np.inf,-np.inf],np.nan);axes[1,col].plot(h.step,100*finite)
        axes[0,col].axhline(level,color='black',ls='--',lw=1);axes[0,col].set_title(f'{int(100*level)}% marginal intervals');axes[0,col].set_ylim(0,1.03);axes[1,col].set_xlabel('Checkups ahead')
        for row in [0,1]:axes[row,col].grid(alpha=.2)
    axes[0,0].set_ylabel('Empirical coverage');axes[1,0].set_ylabel('Mean width (% initial capacity)');fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=5,fontsize=8);fig.tight_layout(rect=[0,.1,1,1]);manifest.append(save(fig,'conformal_coverage_width',['tables/conformal_horizon_overall.csv','tables/conformal_radii.csv']))
    L=pd.read_csv(ROOT/'tables/learning_summary.csv');ref=pd.read_csv(ROOT/'tables/completed_baseline_cells.csv');ref=ref[ref.method.eq('ARIMA_AICc')&ref.origin.eq(20)].APE.mean();fig,ax=plt.subplots(figsize=(7,4.5))
    for method,g in L.groupby('method'):
        g=g.sort_values('fraction');ax.errorbar(100*g.fraction,g.Q_MAPE_equal_cell,yerr=[g.Q_MAPE_equal_cell-g.ci95_low,g.ci95_high-g.Q_MAPE_equal_cell],marker='o',capsize=3,label=method)
    ax.axhline(ref,color='black',ls='--',label='ARIMA');ax.set_xlabel('Fraction of fitting-cell pool (%)');ax.set_ylabel('Equal-cell capacity MAPE (%)');ax.set_xticks([25,50,100]);ax.grid(alpha=.2);ax.legend();fig.tight_layout();manifest.append(save(fig,'training_cell_learning',['tables/learning_summary.csv','tables/completed_baseline_cells.csv']))
    R=pd.read_csv(ROOT/'tables/robustness_summary.csv');fig,axes=plt.subplots(1,3,figsize=(13,4.3),sharey=True)
    for ax,drop in zip(axes,[0,.1,.25]):
        for method,g in R[np.isclose(R.drop_fraction,drop)].groupby('method'):
            g=g.sort_values('noise_sigma_Q0');ax.plot(100*g.noise_sigma_Q0,g.Q_MAPE_equal_cell,marker='o',label=method)
        ax.set_title(f'{int(100*drop)}% prefix checkups missing');ax.set_xlabel('Noise SD (% initial capacity)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Equal-cell capacity MAPE (%)');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=8);fig.tight_layout(rect=[0,.15,1,1]);manifest.append(save(fig,'prefix_noise_missingness',['tables/robustness_summary.csv']))
    O=pd.read_csv(ROOT/'tables/oxford_summary.csv').sort_values('Q_MAPE_equal_cell');fig,ax=plt.subplots(figsize=(9,4.8));ax.bar(np.arange(len(O)),O.Q_MAPE_equal_cell,color='#567d9b');ax.errorbar(np.arange(len(O)),O.Q_MAPE_equal_cell,yerr=[O.Q_MAPE_equal_cell-O.ci95_low,O.ci95_high-O.Q_MAPE_equal_cell],fmt='none',color='black',capsize=3);ax.set_xticks(np.arange(len(O)),O.method,rotation=35,ha='right');ax.set_ylabel('Equal-cell capacity MAPE (%)');ax.set_title('Independent Oxford campaign: eight held-out cells');fig.tight_layout();manifest.append(save(fig,'independent_oxford',['tables/oxford_summary.csv']))
    F=pd.read_csv(ROOT/'tables/validation_mean_frontier.csv');fig,ax=plt.subplots(figsize=(8,5))
    for family,g in F.groupby('family'):
        ax.scatter(g.quiet_latency_cpu1_ms,g.val_Q_MAPE_equal_cell,s=25+35*np.log10(1+g.serialized_bytes/1000),label=family,alpha=.7)
    front=F[F.frontier];ax.scatter(front.quiet_latency_cpu1_ms,front.val_Q_MAPE_equal_cell,s=180,facecolors='none',edgecolors='black',linewidths=1.2,label='Observed inner frontier');ax.set_xscale('log');ax.set_xlabel('Quiet single-thread model-call latency (ms)');ax.set_ylabel('Mean inner-validation capacity MAPE (%)');ax.set_title('Random-search candidates; marker area indicates serialized size');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();manifest.append(save(fig,'inner_accuracy_cost_frontier',['tables/validation_mean_frontier.csv']))
    pd.DataFrame(manifest).to_csv(ROOT/'tables/figure_manifest.csv',index=False)
    (ROOT/'audit/plots_receipt.json').write_text(json.dumps(dict(passed=True,figures=len(manifest),formats=['PNG','PDF'],scope='Data-drivenexports; visual inspection remainsrequiredbeforefinaldelivery. Infiniteinterval widths omittedfromfinitewidth curves; detailedfinitefractioninCSV.'),indent=2),encoding='utf8');print('RESEARCH_FIGURES_EXPORTED',len(manifest),flush=True)
if __name__=='__main__':main()
