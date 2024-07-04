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

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

save_figs= True
use_rare = True

if use_rare: otuFile = 'Data/df.16S_rare_OTUtable.csv'; tag = '_rarefied'
else: otuFile = 'Data/df.16S_OTUtable.csv'; tag = ''

asv_df_obs = pd.read_csv(otuFile, index_col=0)
asv_df_obs = asv_df_obs.loc[asv_df_obs.sum(axis=1) >0]
print('Total dataset: {} sequences, {} taxa'.format(asv_df_obs.sum().sum(), asv_df_obs.shape[0]))
asv_df_obs.index.name = 'asv_id'

meta_file = 'Data/metadata_16S_hydrograph.csv'
meta_df = pd.read_csv(meta_file, parse_dates=True, index_col = '#Sample ID')

#%%
# Source environments
sources_in = pd.read_csv('Data/ProkAtlas_ASV_97_likelihoods.csv', header=[0], index_col = [1])/100
sources_comp = sources_in.drop(labels = [c for c in sources_in.columns if 'Unnamed' in c ], axis = 1)
cols = [sources_comp.columns[i] for i in np.arange(8) if i!=4]
cols.append(sources_comp.columns[4])
sources_comp = sources_comp[cols]
sources = sources_comp.stack([0])
print(sources.groupby(level=[0]).sum()) 

# Determine groups
asv_df2 = asv_df_obs.copy()
asv_df2.columns = asv_df2.columns.map(meta_df.date)
asv_df2.columns = pd.to_datetime(asv_df2.columns)
comm_comp = asv_df2.divide(asv_df2.sum().values) # converts composition to percent
print(comm_comp.sum())

#%% 
sourced_asvs = sources.index.get_level_values(0).unique()

asv_sources = {}
for asv in sourced_asvs:
    asv_sources_frames = []
    if asv in comm_comp.index:
        for day in asv_df2.columns:
            asv_sources_frames.append(pd.DataFrame(sources[asv].mul(comm_comp.loc[asv, day]), columns = [day]))
        res = pd.concat(asv_sources_frames, axis = 1)
        asv_sources[asv] = res
    else: pass
source_frame_all = pd.concat(asv_sources.values(), keys = asv_sources.keys(), names = ['ASV_IDs', 'categories'])
print(source_frame_all.groupby(level=[0]).sum().sum())

#%%

axisfont = 10
tickfont = 6
legendfont = 8

'''Figure 5b Source predictions over the course of the storm (ALL 620 sourced taxa)'''

from matplotlib.legend_handler import HandlerTuple
box_cols = ['00b4d8', '90e0ef']

storm_groups_all = source_frame_all.groupby('categories', level =1).sum()

storm_periods_all = pd.concat([storm_groups_all.loc[:, (storm_groups_all.columns < '10-Oct-2020')], storm_groups_all.loc[:, (storm_groups_all.columns >= '10-Oct-2020') & (storm_groups_all.columns <= '14-Oct-2020')], storm_groups_all.loc[:, storm_groups_all.columns > '14-Oct-2020']], axis = 1, keys = ['pre-event', 'event', 'post-event'])
preMeans_all = storm_periods_all['pre-event'].mean(axis = 1)
storm_changes_all = storm_periods_all[['event','post-event']].sub(preMeans_all, axis = 0).divide(preMeans_all, axis =0)
storm_changes_all = storm_changes_all.reindex(['freshwater', 'sediment', 'other', 'groundwater', 'marine', 'sewage-wastewater', 'biofilm', 'soil'])

sig_df = pd.DataFrame(index = storm_changes_all.index)
dur_p_list, post_p_list, dur_medians, post_medians = [], [], [], []
for i in range(len(storm_changes_all.index)): 
    senv = storm_periods_all.index[i]
    dur_p_list.append(stats.mannwhitneyu(storm_periods_all.loc[senv]['event'], storm_periods_all.loc[senv]['pre-event'])[1])
    post_p_list.append(stats.mannwhitneyu(storm_periods_all.loc[senv]['post-event'], storm_periods_all.loc[senv]['pre-event'])[1])
    dur_medians.append(storm_changes_all.loc[senv]['event'].median())
    post_medians.append(storm_changes_all.loc[senv]['post-event'].median())
sig_df['dur_p'] = dur_p_list
sig_df['dur_median'] = dur_medians
sig_df['post_p'] = post_p_list
sig_df['post_median'] = post_medians
sig_df['i'] = range(len(storm_changes_all.index))

storm_changes_all.rename({'sewage-wastewater': 'sewage-\nwastewater'}, inplace=True)
storm_periods_df_all = storm_changes_all.stack().stack().reset_index()

plt.figure(figsize = (5.2, 3.5))
ax = sns.boxplot(data = storm_periods_df_all, y = 'categories', x= 0,  hue = 'level_2', palette= "Blues_r",
                 linewidth = 0.5, boxprops = {'alpha' : 1.0}, fliersize = 0, zorder = 2)
ax.tick_params(axis='both', which='major', labelsize=8)
plt.xlabel('Normalized difference from mean pre-event\ncommunity representation', fontsize = axisfont)
plt.grid(visible=True, which='major', axis='x', alpha = 0.33)
plt.ylabel('Microbial source environments', fontsize = axisfont)
ax.set_axisbelow(True)

for senv in sig_df.index:
    if sig_df.loc[senv]['dur_p']<0.05: 
        plt.text(np.median(storm_changes_all.loc[senv]['event']), sig_df.loc[senv]['i']-0.1, '**', 
                 fontsize = 8, color = 'k', va= 'center', horizontalalignment='center')
    if sig_df.loc[senv]['dur_p']>0.05 and sig_df.loc[senv]['dur_p']<0.1:
        plt.text(storm_changes_all.loc[senv]['event'].median(), sig_df.loc[senv]['i']-0.1, '*', 
                 fontsize = 8, va = 'center', horizontalalignment='center')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=[handles[0], handles[1]],
  labels=['during-event ($n=$5)', 'post-event ($n=$8)'], handlelength=2,
  handler_map={tuple: HandlerTuple(ndivide=None)}, 
  loc='upper right', fontsize = legendfont)

if save_figs: 
    plt.tight_layout()
    plt.savefig('Figures/FIG_05{}.jpg'.format(tag), dpi = 800)
