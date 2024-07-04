# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:49:20 2022

@author: dawnurycki
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import stats
import seaborn as sns
import statsmodels.stats.multitest as smt
import skbio

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

save_figs= True

dataFile = 'Data/MarysRiver_Hydrograph_Data.csv'
df = pd.read_csv(dataFile)

no_shuffles = 3
use_rare = True

if use_rare: otuFile = 'Data/df.16S_rare_OTUtable.csv'; tag = '_rarefied'
else: otuFile = 'Data/df.16S_OTUtable.csv'; tag = ''

asv_df_obs = pd.read_csv(otuFile, index_col=0)
asv_df_obs = asv_df_obs.loc[asv_df_obs.sum(axis=1) >0]
print('Total dataset: {} sequences, {} taxa'.format(asv_df_obs.sum().sum(), asv_df_obs.shape[0]))
asv_df_obs.index.name = 'asv_id'

meta_file = 'Data/metadata_16S_hydrograph.csv'
meta_df = pd.read_csv(meta_file, parse_dates=True, index_col = '#Sample ID')

# Taxonomy
tax = pd.read_csv('Data/taxonomy_20210924.csv', index_col = 0, delimiter = '\t')
tax.loc[tax.phylum=='p__Actinobacteriota', 'group'] = 'Actinobacteriota'
tax.loc[tax.phylum=='p__Bacteroidota', 'group'] = 'Bacteroidota'
tax.loc[tax.phylum=='p__Cyanobacteria', 'group'] = 'Cyanobacteria'
tax.loc[tax['class']=='c__Gammaproteobacteria', 'group'] = 'Gammaproteobacteria'
tax.loc[tax['class']=='c__Alphaproteobacteria', 'group'] = 'Alphaproteobacteria'
tax.loc[tax.phylum=='p__Verrucomicrobiota', 'group'] = 'Verrucomicrobiota'
tax.loc[tax.group.isnull(), 'group'] = 'Other'
groups = list(tax.group.unique())

q_df = pd.read_csv(dataFile) 
discharge_CFS = q_df['DISCHARGE [CFS]'].values
cfs_to_cms = 0.028316873266469
discharge_cms = discharge_CFS * cfs_to_cms
obsDays = list(meta_df.loc[asv_df_obs.columns].day.astype(int))
q_days = discharge_cms[obsDays]

groups_pivot_frames = []
taxa_pivot_frames =[]

for i in list(range(no_shuffles)):
    if i>0:
        print("Shuffle ", i)
        asv_shuffled = asv_df_obs.copy()
        [np.random.shuffle(v) for k, v in asv_shuffled.items()] # shuffle counts by day
        asv_df = asv_shuffled.sample(frac=1, axis=1)
        asv_df.columns = asv_df_obs.columns
        make_plots = False
    else: 
        asv_df = asv_df_obs
        make_plots = True
    
    asv_list = []
    qcMat = []#np.zeros((asv_df.shape[0],4))*np.nan
    
    startPt = 4# Starting at regression limb if 5, starting at storm is 4
    
    cQ = q_days[startPt:]
    lnQ = np.log10(cQ) # array of discharge values for each day
    
    for b in np.arange(asv_df.shape[0]):
        
        cB = asv_df.iloc[b].values.astype(float)[startPt:] # array of asv counts for each day
        lnB = np.log10(cB) 
        #print(cB)
        
        lnQ2 = lnQ[cB>0] # Discharge on days asv >0
        lnB2 = lnB[cB>0] # asv counts >0
        
        if len(lnB2)>=3: # 3 points to fit regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(lnQ2,lnB2)
            qcLine = [np.nanmean(cB), slope, p_value, np.nan, b]
            asv_list.append(asv_df.iloc[b].name)
            qcMat.append(qcLine)
        #else: print('Dropping {}: <3 points'.format(b))
    
    
    qcMat = np.array(qcMat)
    qc_asv_df = pd.DataFrame(qcMat, index = asv_list, columns = ['Mean_abund', 'slope', 'p', 'p_adj', 'no.'])
    # print(qc_asv_df.shape)
    
    yLimLim = np.ceil(np.nanmax(np.abs(qc_asv_df['slope'])))
    p_adj = smt.multipletests(qc_asv_df['p'], alpha=0.1, method='bonferroni') # Do we need Bonferroni?
    qc_asv_df['p_adj'] = p_adj[1]
    
    meets_pAdj = (qc_asv_df['p']>0) & (qc_asv_df['p']<2) # Why choose 2 here?
    
    qc_asv_df_adj = qc_asv_df.loc[meets_pAdj]
    
    a1=0.1; a2 = 0.05; a3 = 0.01
    inda1 = (qc_asv_df_adj['p']<a1) & (qc_asv_df_adj['p']>a2) # Why not using adjusted p here?
    inda2 = (qc_asv_df_adj['p']<a2) & (qc_asv_df_adj['p']>a3)
    inda3 = (qc_asv_df_adj['p']<a3 )
    nonsig = qc_asv_df_adj.loc[qc_asv_df_adj['p']>a1].index
    
    # Determine groups
    asv_df2 = asv_df.copy()
    asv_df2.columns = asv_df2.columns.map(meta_df.date)
    asv_df2.columns = pd.to_datetime(asv_df2.columns)
    comm_comp = asv_df2.divide(asv_df2.sum().values/100) # /100 converts composition to percent
    comm_comp = pd.concat([comm_comp, qc_asv_df['slope']], axis = 1, join = 'outer', sort=False) # 'outer' includes all 620 otus
    comm_comp.loc[comm_comp.slope>0, 'cqcode'] = 1
    comm_comp.loc[comm_comp.slope<0, 'cqcode'] = -1
    #comm_comp.loc[comm_comp.slope.isnull(), 'cqcode'] = 0
    comm_comp.loc[nonsig, 'cqcode'] = 0 
    #print('Total composition:\n', comm_comp.sum()[:-2])
    
    if i ==0: 
        asv_tax = pd.concat([comm_comp, tax.group], axis = 1, join='inner') 
        asv_tax.drop('slope', axis = 1).to_csv('Data/daily_comm_comp.csv')
    
   
    # Group by sequence counts 
    totals = asv_df.sum(axis=1)
    totals = totals.loc[asv_tax.index].rename("counts")
    groups_df = pd.concat([totals, asv_tax.cqcode, asv_tax.group], axis = 1)
    group_tax = groups_df.group
    group_tax_dict = {s:group_tax.str.count(s).sum() for s in group_tax.unique()}
    group_sums = groups_df.groupby('group').sum().counts.sort_values(ascending=True)
    
    mob = asv_tax[asv_tax.cqcode==1]
    mob_grp = mob.groupby('group').sum()
    mob_seqs = groups_df.loc[mob.index].sum().counts
    dil = asv_tax[asv_tax.cqcode==-1]
    dil_grp = dil.groupby('group').sum()
    dil_seqs = groups_df.loc[dil.index].sum().counts
    static = asv_tax[asv_tax.cqcode==0]
    static_grp = static.groupby('group').sum()
    static_seqs = groups_df.loc[static.index].sum().counts
    
    # Export records
    qc_record = pd.concat([qc_asv_df.iloc[:, :-2], comm_comp.cqcode, tax], axis = 1, join= 'inner')
    if i==0: 
        if save_figs: qc_record.to_csv('Figures/QC_record.csv')
    
    groups_df['group_count'] = groups_df.group.map(group_sums)
    groups_df.cqcode.fillna(2, inplace=True)
    groups_df['frac_of_seqs'] = groups_df.counts/groups_df.group_count
    groups_df['frac_of_taxa'] = 1/groups_df.group.map(group_tax_dict)
    groups_pivot = pd.pivot_table(groups_df, index='group', columns = 'cqcode', values='frac_of_seqs', aggfunc='sum')
    groups_pivot = groups_pivot.reindex(group_sums.index)
    
        
    taxa_pivot = pd.pivot_table(groups_df, index='group', columns = 'cqcode', values = 'frac_of_taxa', aggfunc='sum')
    taxa_pivot = taxa_pivot.reindex(group_sums.index)
    
    if i == 0:
        groups_pivot_obs = groups_pivot
        taxa_pivot_obs = taxa_pivot
    else:
        groups_pivot_frames.append(groups_pivot)
        taxa_pivot_frames.append(taxa_pivot)
    
    if make_plots:
        '''FIGURE 3. C -Q Plots'''
        plt.figure(3)
        plt.title('C-Q slopes for Oct %d to Oct %d in the Marys River\n where C is the relative abundance of microbial ASVs' %(obsDays[startPt], obsDays[-1]) )
        plt.plot(qc_asv_df_adj['no.'],qc_asv_df_adj['slope'],'.',color='gray')
        plt.plot(qc_asv_df_adj[inda1]['no.'],qc_asv_df_adj[inda1]['slope'],'ko',label='p<%.2f (%d total)'%(a1, np.sum(inda1+inda2+inda3)), mfc='y')
        plt.plot(qc_asv_df_adj[inda2]['no.'],qc_asv_df_adj[inda2]['slope'],'ks',label='p<%.2f (%d total)'%(a2, np.sum(inda2+inda3)), mfc='b')
        plt.plot(qc_asv_df_adj[inda3]['no.'],qc_asv_df_adj[inda3]['slope'],'kD',label='p<%.2f (%d total)'%(a3, np.sum(inda3)), mfc='r')
        plt.legend()
        plt.xlabel('Rank Abundance []')
        plt.ylabel('          Dilution ($b$<0)         Mobilizatoin ($b$>0)\n ln[C]-ln[Q]  slope $b$ (n $\geq$ 3)') 
        plt.ylim((-yLimLim,yLimLim)); plt.grid()
        plt.tight_layout()
        plt.show()
        
        '''Figure 4 C-Q classification'''
        axisfont = 10
        tickfont = 6
        legendfont = 8
      
        colors = ["31b1ed","ccc9e7","ffb140","b0a990","a24936","52a937","5f0a87","1b2cc5"]
        colors = ['#{}'.format(c) for c in colors]
        group_colors = dict(zip(groups, colors))
        
        fig, axs = plt.subplots(1, 3, figsize = (7, 3.5) )
        
        mob_grp.iloc[:, :-2].T.plot.bar(stacked='True', ax = axs[0], color = [group_colors[g] for g in mob_grp.index], legend=0, width=0.9,lw=0.0)
        dil_grp.iloc[:, :-2].T.plot.bar(stacked='True', ax = axs[2], color = [group_colors[g] for g in dil_grp.index], legend=0, width=0.9,lw=0)
        static_grp.iloc[:, :-2].T.plot.bar(stacked='True', ax = axs[1], color = [group_colors[g] for g in static_grp.index], legend=0, width=0.9,lw=0)
        if i == 0: 
            print ('Avg pre-event mean proportion of community:\n',
                   'Mobilized: %.2f; Diluted: %.2f' 
                   %(mob_grp.loc[:, pd.to_datetime('10/6/2020'):pd.to_datetime('10/9/2020')].sum().mean(), 
                     dil_grp.loc[:, pd.to_datetime('10/6/2020'):pd.to_datetime('10/9/2020')].sum().mean()) )
            print ('Peak event discharge proportion of community:\n',
                   'Mobilized: %.2f; Diluted: %.2f' 
                   %(mob_grp[pd.to_datetime('10/14/2020')].sum(), 
                     dil_grp[pd.to_datetime('10/14/2020')].sum()))
        for ax in fig.axes:
            ax.set_xticklabels([x.day for x in mob.columns[:17]], fontsize = tickfont, rotation = 90)
            ax.tick_params(axis = 'both', labelsize = tickfont)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        
        axs[0].set_title('Mobilized\n({} seq, {} taxa)'.format(mob_seqs, len(mob)), y=1.0, pad=5, fontsize = axisfont)
        axs[1].set_title('Static\n({} seq, {} taxa)'.format(static_seqs, len(static)), y=1.0, pad=5, fontsize = axisfont)
        axs[2].set_title('Diluted\n({} seq, {} taxa)'.format(dil_seqs, len(dil)), y=1.0, pad=5, fontsize = axisfont)
        #axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axs[1].set_xlabel('Day in October', fontsize = axisfont)
        axs[0].set_ylabel('Community composition (%)', fontsize=axisfont)
        
        plt.tight_layout()
        plt.subplots_adjust(wspace = 0.16, bottom = 0.25)
        
        handles, labels = axs[1].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol = 4, fontsize = legendfont, fancybox=True)
        
        if save_figs: 
            plt.savefig('Figures/FIG_03{}.jpg'.format(tag), dpi = 800)

    
    
        ''' FIGURE 4. Characterize Phylogentic Groups '''
        barcolors = {"Dodger Blue": '1F9EFF', "Naples Yellow":"f5d451","Dark Pastel Green":"0bc148","Light Gray":"d1d1d1","Spanish Carmine":"d11149", "Blue RYB":"2a4edf",}
        barcolors = ['#{}'.format(c) for c in barcolors.values()]
        fig, axs = plt.subplots(1, 2, sharey = True, figsize=(7.0, 3))            
        ax1 = groups_pivot.plot.barh(stacked=True, ax = axs[0], color=barcolors, width = 0.75)
        ax2 =  taxa_pivot.plot.barh(stacked=True, ax = axs[1], color=barcolors, width = 0.75)
        for i in range(len(groups_pivot.index)):
            grp = groups_pivot.index[i]
            ax1.text(0.02, i, '{} seq.'.format(str(group_sums[i])), color='indigo', 
                     fontstyle ='italic', va='center', fontsize = 8)
            ax2.text(0.78, i, '{} taxa'.format(str(group_tax_dict[grp])), color = 'indigo',
                     fontstyle = 'italic', va = 'center', fontsize = 8)
        
        axs[0].set_xlim(0, 1.0)
        axs[1].set_xlim(0, 1.0)
        
        h, lab = plt.gca().get_legend_handles_labels()
        h, lab = axs[0].get_legend_handles_labels()
        leg_dict = {'-1.0': 'Diluted', '0.0': 'Static', '1.0': 'Mobilized', '2.0': 'Uncharacterized'}
        new_lab = [leg_dict[l] for l in lab]
        plt.legend(h, new_lab, loc = 'lower center', ncol = 4, 
                   bbox_to_anchor=(-0.05,  1.0), fontsize=legendfont, fancybox=True)#, shadow=True)
        axs[0].legend().remove()
        labels = ['${}$'.format(item.get_text()) for item in ax1.get_yticklabels()]
        ax1.set_yticklabels(labels)
        ax1.tick_params(axis='x', labelsize=tickfont)
        ax2.tick_params(axis='x', labelsize=tickfont)
        ax1.set_ylabel('')
        ax1.set_xlabel('Fraction of $n$ sequences', fontsize=axisfont)
        ax2.set_xlabel('Fraction of $m$ taxa', fontsize=axisfont)
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1)
        if save_figs: 
            plt.savefig('Figures/FIG_04{}.jpg'.format(tag), dpi=800)


if no_shuffles>0:
	groups_pivot_all = pd.concat(groups_pivot_frames, axis = 1, join = 'outer', keys = ['run %s' %i for i in list(range(1, no_shuffles+1))], sort=False)

	summ_frames = []
	[summ_frames.append(groups_pivot_all.loc[g].groupby('cqcode').describe(percentiles=[0.05, 0.95])) for g in groups]
	summary = pd.concat(summ_frames, axis = 1, keys = groups)

	groups_pivot_shuffled_97 = summary.xs('95%', level=1, axis = 1).T
	groups_pivot_97 = groups_pivot_shuffled_97.reindex(groups_pivot_obs.index)
	groups_pivot_shuffled_2 = summary.xs('5%', level=1, axis = 1).T
	groups_pivot_2 = groups_pivot_shuffled_2.reindex(groups_pivot_obs.index)

	mask_frames=[]
	for g in groups_pivot_obs.index:
	    cur_g = groups_pivot_obs.loc[g].between(groups_pivot_2.loc[g], groups_pivot_97.loc[g])
	    mask_frames.append(cur_g)
	sig_mask = pd.concat(mask_frames, axis = 1).T  

	groups_pivot_obs[~sig_mask] 
print('\nDone.')
