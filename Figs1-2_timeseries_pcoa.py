#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:51:55 2021

@author: dawnurycki, stephengood
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import stats
import skbio
import os
import statsmodels
from datetime import datetime, timedelta

save_figs= False
dataFile = 'Data/MarysRiver_Hydrograph_Data.csv'

use_rare = True
tag = '_rarefied' if use_rare== True else ''

try: df = pd.read_csv(dataFile)
except: 
    os.chdir(os.path.dirname(__file__))
    df = pd.read_csv(dataFile)
    
basin_area = 156 # squre miles
cfs_to_cms = 0.028316873266469
squaremiles_to_squrekm = 2.58999
basin_area_m2 = basin_area * squaremiles_to_squrekm * 1000**2

day = df['DAY'].values
d2h_df = df['d2HVSMOW (‰)']
d2h = d2h_df.values
d2h_df.dropna(inplace=True)#.values
d18o_df = df['d18OVSMOW (‰)'].dropna()#.values
iso_days = df['DAY'][df['d18OVSMOW (‰)']==df['d18OVSMOW (‰)']].values
discharge_CFS = df['DISCHARGE [CFS]'].values
discharge_df_cms = df['DISCHARGE [CFS]']* cfs_to_cms
iso_discharge_cms = df['DISCHARGE [CFS]'][df['d18OVSMOW (‰)']==df['d18OVSMOW (‰)']].values * cfs_to_cms

discharge_cms = discharge_CFS * cfs_to_cms
flow = discharge_CFS * cfs_to_cms * (60*60*24)/(basin_area_m2) * 1000
ppt = df['ppt (mm)']

d2h_pre_3 = -38.7; d18o_pre_3 = -6.3; pre_3_days = np.array([-1,13])
d2h_pre_4 = -23.3; d18o_pre_4 = -5.0; pre_4_days = np.array([14,33])
dex_pre_3 = d2h_pre_3 - 8 * d18o_pre_3
dex_pre_4 = d2h_pre_4 - 8 * d18o_pre_4

d2h_rel_to_pre = np.array([d2h - d2h[5],d2h - d2h[6],d2h - d2h[7],d2h - d2h[8]])
d2h_base = np.nanmean(d2h[day<=9])
d2h_base = np.nanmean(d2h[day<=9])


## Sample baseflow separation ###
print('Pre-event d2H =', np.mean(d2h_df[:4]))
print('Precip d2H =', d2h_pre_3)
print('13 Oct, stream d2H =', d2h_df[13])


otuFile = 'Data/df.16S_OTUtable.csv'
meta_file = 'Data/metadata_16S_hydrograph.csv'
meta_df = pd.read_csv(meta_file, parse_dates=True, index_col = '#Sample ID')

df_isoDat = df[['d2HVSMOW (‰)','d18OVSMOW (‰)']]
df_isoDat.dropna(inplace=True)#.values
df_isoDat = df_isoDat.T

def getDistances(indf, dist = 'BC'):
    dist_df = pd.DataFrame(index = indf.columns, columns = indf.columns)
    for i in range(len(indf.columns)):
        for j in range(len(indf.columns)):
            print(i, j)
            c1 = indf[indf.columns[i]]
            c2 = indf[indf.columns[j]]
            
            if dist == 'BC': dist_df.iloc[i,j] = distance.braycurtis(c1,c2)
            else: dist_df.iloc[i,j] = distance.euclidean(c1,c2)
            
            print(i,j,dist_df.iloc[i,j])
    return dist_df

bc_iso = getDistances(df_isoDat)
bc_iso_reltopre = bc_iso.iloc[[0,1,2,3]]
bc_iso_reltopre_days = bc_iso_reltopre.columns.values
bc_iso_reltopre_mean = np.mean(bc_iso_reltopre,axis=0)
bc_iso_reltopre_stds = np.std(bc_iso_reltopre,axis=0)

def getDiffFromPre(inArray):
    raw_array_reltopre = np.array([inArray - inArray[0],
                                   inArray - inArray[1],
                                   inArray - inArray[2],
                                   inArray - inArray[3]])
    raw_array_reltopre_mean = np.mean(raw_array_reltopre,axis=0)
    raw_array_reltopre_std = np.std(raw_array_reltopre,axis=0)
    return raw_array_reltopre_mean, raw_array_reltopre_std

raw_iso_reltopre_mean, raw_iso_reltopre_std = getDiffFromPre(d2h_df.values)
raw_iso_dex = d2h_df.values - 8 * d18o_df.values
raw_dex_reltopre_mean, raw_dex_reltopre_std = getDiffFromPre(raw_iso_dex)

df = df.T
#%%
df_unrare = pd.read_csv(otuFile)
asv_df_all = df_unrare.set_index('Unnamed: 0').T
asv_df_all = asv_df_all.loc[:,asv_df_all.sum() > 0]
asv_df_all.index = asv_df_all.index.map(meta_df.date)
asv_df_all.set_index(pd.to_datetime(asv_df_all.index), inplace=True)
print(asv_df_all.sum(axis =1))

daily_counts = asv_df_all.sum(axis=1)

rich_unrare = asv_df_all.astype(bool).sum(axis=1)
shan_unrare = stats.entropy(asv_df_all.T, base=2)

rare_val = 4000
asvs = df_unrare['Unnamed: 0']
cols = [c for c in df_unrare.columns if 'GH' in c]


rare_num = 500
rare_tax = []
no_detect = []
no_days_dict = {'1': [], '2': [], '3': [], '4': [], '5': [], '+': []}
rich_ba = []
shan_ba = {'d': [], 'W': [], 'p': []}

for n in range(rare_num): 
    rare_df = pd.DataFrame(index = asvs)
    for col in cols: rare_df[col] = skbio.stats.subsample_counts(df_unrare[col], rare_val)
    
    asv_df_rare= rare_df.T
    asv_df_rare = asv_df_rare.loc[:,asv_df_rare.sum() > 0]
    asv_df_rare.index = asv_df_rare.index.map(meta_df.date)
    asv_df_rare.set_index(pd.to_datetime(asv_df_rare.index), inplace=True)
    #print(asv_df_rare.sum(axis =1))
    
    rich = asv_df_rare.astype(bool).sum(axis=1)
    rich_ba.append(np.median(rich))
    shan = stats.entropy(asv_df_rare.T, base=2)
    shan_ba['d'].append(np.mean(shan_unrare-shan))
    wilcoxon = stats.wilcoxon(shan_unrare - shan)
    shan_ba['W'].append(wilcoxon[0])
    shan_ba['p'].append(wilcoxon[1])
    
    rare_taxa = [*(set(asv_df_all.columns).difference(set(asv_df_rare.columns))), ]
    rare_tax.append(len(rare_taxa))
    no_detect.append(list(asv_df_all[rare_taxa].sum().values))
    rtd = asv_df_all[rare_taxa].astype(bool).sum().value_counts()
    
    for nd in rtd.index:
        pc = rtd[nd]/(len(rare_taxa))
        no_days_dict[str(nd)].append(pc)
        if nd > 5: no_days_dict['+'].append(pc)

print('E.g., for the final rarefaction: \n\n')   
print('Number of unique sequences lost to rarefaciton: ', len(rare_taxa))
print('Median number of dections of rare taxa across all 17 samples: ', np.median(asv_df_all[rare_taxa].sum()))
print('Number of days on which each rare taxa was detected:\n', asv_df_all[rare_taxa].astype(bool).sum().value_counts())
    
print('Range of sample library size: {}-{} reads per sample'.format(daily_counts.min(), daily_counts.max()))

print('\n\nCorrelation between alpha diversity (unrarefied) and sequence counts:')
print('Richness: ',stats.spearmanr(rich_unrare, daily_counts))
print('Shannon: ',stats.spearmanr(shan_unrare, daily_counts))

print('\n\nCorrelation between alpha diversity (rarefied) and sequence counts:')
print('Richness: ',stats.spearmanr(rich, daily_counts))
print('Shannon: ',stats.spearmanr(shan, daily_counts))

print('\n\nCorrelation in alpha diversity between rarefied and unrarefied counts:')
print('Richness: ',stats.spearmanr(rich_unrare, rich))
print('Shannon: ',stats.spearmanr(shan_unrare, shan))

print('Median species richness of unrarefied and rarefied stamples, respectively: {}, {} unique taxa'.format(np.median(rich_unrare), np.mean(rich_ba)))

print('Mean change in median Shannon index of unrarefied and rarefied samples: {}'.format(np.mean(shan_ba['d'])))
print('Wilcoxon paired sample test for difference: W = {:.1f}, p = {:.3f}'.format(np.mean(shan_ba['W']), np.mean(shan_ba['p'])))

print('\n\n\n\n\n')
print('Mean number of unique sequences lost to rarefaction:', np.mean(rare_tax))
print('Mean number of detections per rare taxa:', np.mean(sum(no_detect, [])))
for k,v in no_days_dict.items():
    print("Mean percentage of rare taxa detected on exactly %s day(s) =" %k, np.median(v))

    
    

#%%

if use_rare: 
    asv_df = asv_df_rare
    shannon = shan
    richness = rich
else: 
    asv_df = asv_df_all
    shannon = shan_unrare
    richness = rich_unrare   

# Correlation between discharge, precip, and alpha diversity
shannon_df = pd.Series (shannon, index = richness.index.day)
alpha_Q = pd.concat([discharge_df_cms, ppt, shannon_df, pd.Series(richness.values, index = richness.index.day.to_list())], 
                    axis=1, join = 'inner', keys = ['discharge', 'precip', 'shannon', 'richness'])
lag_Q =[(discharge_df_cms[i-1]) for i in alpha_Q.index]
lag_P =[(ppt[i-1]) for i in alpha_Q.index]
lag_P_2d = [(ppt[i-2]) for i in alpha_Q.index]
alpha_Q['discharge_t-1'] = lag_Q
alpha_Q['precip_t-1'] = lag_P
alpha_Q['precip_t-2'] = lag_P_2d

df_z = asv_df.T.copy()
df_ec = df_z.apply(stats.zscore)
df_ec = df_ec.T

bc_df = getDistances(asv_df.set_index(asv_df.index.day).T)    
ec_df = getDistances(asv_df.set_index(asv_df.index.day).T, dist = 'Euclidean')

#%%        
sample_days = asv_df_rare.index.day


bc_rel_to_pre = bc_df.iloc[:4,:]
bc_rel_to_pre_mean = bc_rel_to_pre.mean(axis=0)
bc_rel_to_pre_std = bc_rel_to_pre.std(axis=0)

print ('\nDifference in community BC:')
print('{:.3f} (SD = {:.3f}) pre-event mean'.format(
    bc_rel_to_pre_mean.loc[6:9].mean(),
    bc_rel_to_pre_mean.loc[6:9].std()))

print('{:.3f} (SD = {:.3f}) pre- vs. post-'.format(
    bc_rel_to_pre_mean.loc[10:].mean(),
    bc_rel_to_pre_mean.loc[10:].std()))

print('{:.3f} ({} Oct) max'.format(
    bc_rel_to_pre_mean.max(), 
    np.argmax(bc_rel_to_pre_mean)))

# Difference in pre- and post-event istopic ratios 
print('\nDifference in 2H:')
print('{:.3f} (SD = {:.3f}) pre-event mean'.format(
    raw_iso_reltopre_mean[0:4].mean(),
    raw_iso_reltopre_mean[0:4].std()))

print('{:.3f} (SD = {:.3f}) pre- vs. post-'.format(
    raw_iso_reltopre_mean[4:].mean(),
    raw_iso_reltopre_mean[4:].std()))

print('{:.3f} ({} Oct) max'.format(
    raw_iso_reltopre_mean.max(), 
    sample_days[np.argmax(raw_iso_reltopre_mean)])) 

print('\n{:.3f}, {:.3f} (Difference between pre-event stream mean and aggregated precip)'.format(
    d2h_pre_3-df_isoDat.iloc[0,:4].mean(),
    d2h_pre_4-df_isoDat.iloc[0,:4].mean()))

print('\nDifference in d-excess:')
print('{:.3f} (SD = {:.3f}) pre-event mean'.format(
    raw_dex_reltopre_mean[0:4].mean(),
    raw_dex_reltopre_mean[0:4].std()))

print('{:.3f} (SD = {:.3f}) pre- vs. post-'.format(
    raw_dex_reltopre_mean[4:].mean(),
    raw_dex_reltopre_mean[4:].std()))

print('{:.3f} ({} Oct) max'.format(
    raw_dex_reltopre_mean.max(), 
    sample_days[np.argmax(raw_dex_reltopre_mean)])) 

print('\n{:.3f}, {:.3f} (Difference between pre-event stream mean and aggregated precip)'.format(
    dex_pre_3-raw_iso_dex[0:4].mean(),
    dex_pre_4-raw_iso_dex[0:4].mean()))

# Discharge
print('\nPearson correlation Discharge vs. Alpha diversity DAYS 7 - 25 OCTOBER:\n Richness {}, \nShannon {}'.format(
    stats.pearsonr(alpha_Q.iloc[2:].discharge, alpha_Q.iloc[2:].richness), 
    stats.pearsonr(alpha_Q.iloc[2:].discharge, alpha_Q.iloc[2:].shannon)))
print('One day lagged: Q(t-1):\n richess {}\n Shannon {}'.format(
    stats.pearsonr(alpha_Q['discharge_t-1'], alpha_Q.richness), 
    stats.pearsonr(alpha_Q['discharge_t-1'], alpha_Q.shannon)))
print('One day lagged: Q(t-1) excluding outlier:\n richess {}\n Shannon {}'.format(
    stats.pearsonr(alpha_Q.iloc[1:]['discharge_t-1'], alpha_Q.iloc[1:].richness), 
    stats.pearsonr(alpha_Q.iloc[1:]['discharge_t-1'], alpha_Q.iloc[1:].shannon)))

# Precip
print('\n\nPearson correlation Precipitation vs. Alpha Diversity DAYS 7 - 25 OCTOBER:\n Richness {}, \nShannon {}'.format(
    stats.pearsonr(alpha_Q.iloc[2:].precip, alpha_Q.iloc[2:].richness), 
    stats.pearsonr(alpha_Q.iloc[2:].precip, alpha_Q.iloc[2:].shannon)))
print('One day lagged:\n Richness {}, \nShannon {}'.format(
    stats.pearsonr(alpha_Q['precip_t-1'], alpha_Q.richness), 
    stats.pearsonr(alpha_Q['precip_t-1'], alpha_Q.shannon)))
print('One day lagged excluding outlier:\n Richness {}, \nShannon {}'.format(
    stats.pearsonr(alpha_Q.iloc[1:]['precip_t-1'], alpha_Q.iloc[1:].richness), 
    stats.pearsonr(alpha_Q.iloc[1:]['precip_t-1'], alpha_Q.iloc[1:].shannon)))
print('Two day lagged excluding outlier:\n Richness {}, \nShannon {}'.format(
    stats.pearsonr(alpha_Q.iloc[1:]['precip_t-2'], alpha_Q.iloc[1:].richness), 
    stats.pearsonr(alpha_Q.iloc[1:]['precip_t-2'], alpha_Q.iloc[1:].shannon)))
   

#%%
#
from sklearn.preprocessing import StandardScaler 
import seaborn as sns; sns.set_style("whitegrid", {'axes.grid' : False})
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

PCoA_dims = 4

# Get SData
DF_data = asv_df
n,m = DF_data.shape
print(n,m)

# Scaling mean = 0, var = 1
DF_standard = pd.DataFrame(StandardScaler().fit_transform(DF_data), 
                           index = DF_data.index,
                           columns = DF_data.columns)

# Distance Matrix
Ar_dist = distance.squareform(distance.pdist(DF_data, metric="braycurtis")) # (n x n) distance measure
PCoA = skbio.stats.ordination.pcoa(Ar_dist,number_of_dimensions=17)

samples = PCoA.samples
print(samples.shape)
print(PCoA.proportion_explained)
xx = samples.values[:,0]
yy = samples.values[:,1]

import statsmodels.api as sm
def doReg(inComs,ifPrint):
    Xdata = samples.iloc[:,:inComs]
    #yObs = np.expand_dims(iso_discharge_cms,axis=1)
    yObs = iso_discharge_cms
    
    Xdata = sm.add_constant(Xdata)
    model = sm.OLS(yObs,Xdata)
    results = model.fit()
    params = results.params
    ypred = results.predict(Xdata)
    rsq = results.rsquared
    adjrsq = results.rsquared_adj
    if ifPrint: print(results.summary())
    return ypred, rsq, adjrsq

ypred_PCoA_dims, yrsq_PCoA_dims, adjrsq_PCoA_dims = doReg(PCoA_dims, True)

#%%  Other Regressions

from scipy import stats
y = iso_discharge_cms
x1 = rich.values
res1 = stats.linregress(x1, y)

x2 = shannon
res2 = stats.linregress(x2, y)


cnt = np.arange(1,18)
rsq_lst = np.copy(cnt)*0.0
adj_rsq_lst = np.copy(cnt)*0.0

for c in cnt:
    ypred, yrsq, yadjrsq = doReg(c, True)
    rsq_lst[c-1] = yrsq
    adj_rsq_lst[c-1] = yadjrsq


rich_pred = x1*res1.slope + res1.intercept
shannon_pred = x2*res2.slope + res2.intercept


yObs = np.expand_dims(iso_discharge_cms,axis=1)
iso_dat = np.array([d2h_df.values,d18o_df.values]).T
iso_dat_wc = sm.add_constant(iso_dat)
iso_model = sm.OLS(yObs,iso_dat_wc)
iso_results = iso_model.fit()
iso_params = iso_results.params
iso_pred = iso_results.predict(iso_dat_wc)

#%%

legFontSize = 8
axFontSize = 6
labFontSize = 10
malpha = 1.0
msize = 4

#%% FIGURE 1

'''Figure 1'''
xpos = 5.5; ypos = 0.875
# F1A

colors = ["1b9e77","d95f02","7570b3","e7298a","e6ab02","66a61e","666666","a6761d", "660066"]
colors = ['#{}'.format(c) for c in colors]

fig, axs = plt.subplots(4, 1, sharex=True, figsize=(3.43,7.8))

# Alpha diversity
rcolor = colors[0]
scolor = colors[1]
axs[1].plot(rich.index.day, rich.values, color = rcolor, marker = 's', 
            lw = 0.5, markersize = msize, alpha = malpha, label = "Richness") # s = msize, (for scatter)
axs[1].set_ylabel('Microbial richness (# taxa)', color = rcolor, fontsize = labFontSize)
axs[1].text(xpos, 300*ypos, 'B) Microbial\ndiversity',fontsize=legFontSize)
axs[1].set_yticks([0,100,200,300])
axs[1].set_ylim((0,300))
axs[1].tick_params(axis='y',labelsize = axFontSize)


axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].spines['bottom'].set_visible(False)
axs[1].xaxis.grid(alpha = 0.5)

ax0b = axs[1].twinx()
ax0b.plot([], color = rcolor, marker = 's', 
            lw = 0.5, markersize = msize, alpha = malpha, label = "Richness")
ax0b.plot(rich.index.day, shannon, color = scolor, marker = 'p', 
         lw = 0.5, markersize = msize, alpha = malpha, label = "Shannon\nindex") # s = msize
ax0b.set_ylabel('Shannon index (-)', color = scolor, fontsize = labFontSize)
ax0b.set_ylim(3, 6)
ax0b.set_yticks([3.0,4.0,5.0,6.0])
ax0b.spines['left'].set_visible(False)
ax0b.spines['top'].set_visible(False)
ax0b.spines['bottom'].set_visible(False)
ax0b.xaxis.grid(0.5)
ax0b.tick_params(axis='y',labelsize = axFontSize)

# F1B

pre_lw = 4
# 2H
ccolor = colors[2]
axs[0].plot(pre_3_days, pre_3_days*0+d2h_pre_3, color = ccolor, label='Precipitation',
            lw = pre_lw,alpha = 0.75)
axs[0].plot(iso_days, d2h_df, marker = '^', color = ccolor, label='Stream', 
            lw = 0.5, markersize = msize, alpha = malpha) # s= msize
axs[0].plot(pre_4_days, pre_4_days*0+d2h_pre_4, color = ccolor, alpha = 0.75,lw = pre_lw,)
axs[0].set_ylabel(r'Sample $\delta$$^2H$ (‰)', color = ccolor, fontsize = labFontSize); #plt.legend()
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['bottom'].set_visible(False)
axs[0].xaxis.grid(alpha = 0.5)
axs[0].set_yticks([-60,-50,-40,-30,-20])
axs[0].tick_params(axis='y',labelsize = axFontSize)
axs[0].text(xpos, -60+40*ypos, 'A) Stable\nwater isotopes',fontsize=legFontSize)

# 18O
ccolor = colors[3]
ax1b = axs[0].twinx()
ax1b.plot(pre_3_days, pre_3_days*0+d18o_pre_3, color = ccolor, label='Precipitation', 
          lw = pre_lw,alpha = 0.75)
ax1b.plot(iso_days,d18o_df, marker = 'v', color = ccolor, 
         lw = 0.5, markersize = msize, alpha = malpha) # s = msize, 
ax1b.plot([], marker = '^', color = 'k', label='Stream', 
            lw = 0.5, markersize = msize) # s= msize
ax1b.plot(pre_4_days, pre_4_days*0+d18o_pre_4, color = ccolor, alpha = 0.75,lw = pre_lw)

leg = ax1b.legend(fontsize = axFontSize, loc = 'lower right', ncol = 2, frameon = False)
leg.get_frame().set_linewidth(0.0)
leg.legendHandles[0].set_color('k') 
ax1b.set_ylabel(r'Sample $\delta$$^{18}O$ (‰)', color = ccolor, fontsize = labFontSize); #plt.legend()
ax1b.set_yticks([-9,-8,-7,-6,-5])
ax1b.spines['left'].set_visible(False)
ax1b.spines['top'].set_visible(False)
ax1b.spines['bottom'].set_visible(False)
ax1b.tick_params(axis='y',labelsize = axFontSize)


offset = 0.0
capsz = 0#0.1
elw = 0#.05

axs[2].scatter(bc_iso_reltopre_days+1-offset, raw_iso_reltopre_mean, marker= '<',
                color=colors[5], facecolors = 'none', s = msize*3.5, alpha = malpha, label = '$\delta$$^2H$')

axs[2].scatter(bc_iso_reltopre_days+1+offset, raw_dex_reltopre_mean, marker = '>',
                edgecolor=colors[5], facecolors = colors[5], s = msize*3.5, alpha = malpha, label = '$d$-$excess$')

axs[2].plot()
axs[2].set_ylabel('Isotopic difference (‰)', color=colors[5], fontsize = labFontSize)
lge = axs[2].legend(fontsize = axFontSize, loc = 'lower right', frameon = False).set_zorder(12)
leg.get_frame().set_linewidth(0.0)

axs[2].spines['left'].set_visible(False)
axs[2].spines['top'].set_visible(False)
axs[2].spines['bottom'].set_visible(False)
axs[2].xaxis.grid(alpha = 0.5)
axs[2].tick_params(axis='y',labelsize = axFontSize)
axs[2].tick_params(axis='x',labelsize = axFontSize)
axs[2].set_ylim((-2,4))
axs[2].text(xpos, -2+6*ypos, 'C) Average distance metrics relative\nto pre-event conditions (days 6,7,8,9)',fontsize=legFontSize)

malpha2b = 1.0
ax2b = axs[2].twinx()
ax2b.errorbar(bc_rel_to_pre_mean.index[4:], bc_rel_to_pre_mean.values[4:],yerr=bc_rel_to_pre_std[4:], 
              fmt='D', mec=colors[4], mfc= 'none', mew = 0.75, ecolor=colors[4], elinewidth=0.75, capsize=0, 
              markersize = msize, alpha = malpha2b, label = "post-event")
ax2b.errorbar(bc_rel_to_pre_mean.index[:4], bc_rel_to_pre_mean.values[:4],yerr=bc_rel_to_pre_std[:4], 
              fmt='D', mec=colors[4], mfc= 'none', mew = 0.75, ecolor=colors[4], elinewidth=0.75, capsize=0, 
              markersize = msize, alpha = malpha2b, label = 'pre-event')

 
ax2b.set_ylabel('Microbial dissimilarity (-)', 
                  color=colors[4], fontsize = labFontSize)
ax2b.set_yticks([0,0.1,0.2,0.3,0.4,0.5])
ax2b.spines['right'].set_visible(False)
ax2b.spines['top'].set_visible(False)
ax2b.spines['bottom'].set_visible(False)
ax2b.xaxis.grid(alpha = 0.5)
ax2b.tick_params(axis='y',labelsize = axFontSize)


axs[3].fill_between(day,0+0*discharge_cms,discharge_cms, color = colors[6], lw = 0.5, alpha = malpha)
axs[3].plot(iso_days,ypred_PCoA_dims,'k-',label='%d coordinate\nPCoA reg.' %PCoA_dims)
axs[3].plot(iso_days,rich_pred,'k--',label='Richnesss reg.' )
axs[3].plot(iso_days,shannon_pred,'k:',label='Shannon reg.' )

axs[3].legend(fontsize = axFontSize, loc = 'upper right',frameon=False).set_zorder(12)
axs[3].set_ylabel('Marys River discharge (cms)', color=colors[6], fontsize = labFontSize)
axs[3].set_xlabel('Day in October 2020', fontsize = labFontSize)
axs[3].set_ylim(0, 2)
axs[3].spines['right'].set_visible(False)
axs[3].spines['top'].set_visible(False)
axs[3].xaxis.grid(alpha = 0.5)
axs[3].tick_params(axis='y',labelsize = axFontSize)
axs[3].tick_params(axis='x',labelsize = axFontSize)
axs[3].set_yticks([0,0.5,1.0,1.5,2.0,2.5])
axs[3].text(xpos, 2.5*ypos, 'D) Hydrologic\nfluxes',fontsize=legFontSize)

ax3b = axs[3].twinx()
ax3b.bar(day,ppt,color=colors[7],lw=0)
ax3b.set_ylabel('Daily precipitation (mm)', 
                   color=colors[7], fontsize = labFontSize)
ax3b.set_yticks([0,5,10,15,20])
ax3b.spines['top'].set_visible(False)
ax3b.tick_params(axis='y',labelsize = axFontSize)

ax4b = axs[3].twinx()
ax4b.plot(iso_days,ypred_PCoA_dims,'k-',label='%d coordinate\nPCoA reg.' %PCoA_dims)
ax4b.plot(iso_days,rich_pred,'k--',label='Richnesss reg.' )
ax4b.plot(iso_days,shannon_pred,'k:',label='Shannon reg.' )
ax4b.set_yticks([0,0.5,1.0,1.5,2.0,2.5])
ax4b.spines['left'].set_visible(False)
ax4b.spines['top'].set_visible(False)
ax4b.set_axis_off()

plt.xlim(5,26)
plt.subplots_adjust(hspace = -1.0)
plt.tight_layout()
fig.align_ylabels()

if save_figs: 
    plt.savefig('Figures/FIG_01{}.jpg'.format(tag), dpi = 800)

#%% FIGURE 2

f1 = plt.figure(2,figsize=(7,3.5))
ax = plt.subplot(111)

plt.plot(xx,yy,':',zorder=0, color='gray')
plt.scatter(xx, yy, c=iso_discharge_cms, 
            s=msize*20, lw=1, label="NMDS",zorder=2,edgecolor='k',cmap='gist_stern')
cbar = plt.colorbar(pad=.02,shrink=0.75); plt.clim((0,2.0))
cbar.set_label(label='Marys River discharge (cms)', fontsize = labFontSize)

ax.set_xlabel('PCoA coordinate 1\n(proportion explained = %.3f)'%PCoA.proportion_explained[0],
              fontsize = labFontSize); #plt.legend()
ax.set_ylabel('PCoA coordinate 2\n(proportion explained = %.3f)'%PCoA.proportion_explained[1],
              fontsize = labFontSize); #plt.legend()
ax.tick_params(axis='x',labelsize = axFontSize)
ax.tick_params(axis='y',labelsize = axFontSize)
if use_rare==False: ax.set_ylim(-0.22, 0.32)
try: cbar.set_ticks([0,0.5,1,1.5,2], fontsize = axFontSize)
except: cbar.set_ticks([0,0.5,1,1.5,2])
cbar.ax.tick_params(labelsize=axFontSize)


txt_spacer = 0.004
for p in np.arange(len(xx)):
    if (iso_days[p] == 19)|(iso_days[p] == 17)|(iso_days[p] == 8)| (iso_days[p] == 11)| (iso_days[p] == 25): # Up and to left
        plt.text(xx[p]-txt_spacer, yy[p]+txt_spacer,'%d'%iso_days[p],
                 horizontalalignment='right',verticalalignment='bottom')        
    elif (iso_days[p] == 21) | (iso_days[p] == 23)|(iso_days[p] == 9): # Down and to left
        plt.text(xx[p]-txt_spacer, yy[p]-txt_spacer,'%d'%iso_days[p],
                 horizontalalignment='right',verticalalignment='top')     
    elif  (iso_days[p] == 18) | (iso_days[p] == 12)| (iso_days[p] == 10): # Down and to right
        plt.text(xx[p]+txt_spacer, yy[p]-txt_spacer,'%d'%iso_days[p],
                 horizontalalignment='left',verticalalignment='top')     
    else:    
        plt.text(xx[p]+txt_spacer, yy[p]+txt_spacer,'%d'%iso_days[p])


inSetSize = 0.35

if use_rare: axins = ax.inset_axes(bounds = [0.70, 0.15,inSetSize,inSetSize])
else: axins = ax.inset_axes(bounds = [0.70, 0.65,inSetSize,inSetSize])
axins.plot(iso_discharge_cms,ypred_PCoA_dims,'o',color='gray',markersize=msize,zorder=2)
axins.plot([0,np.max(iso_discharge_cms)],[0,np.max(iso_discharge_cms)],'k-' ,zorder=1); tick_labs = [0.0,0.5,1.0,1.5,2.0]
axins.set_aspect('equal')
axins.text(0.1,1.6,'%d coordinate\nPCoA reg.'%(PCoA_dims),fontsize = axFontSize)
axins.text(0.75,0.3,'Adj. $r^2$=%.2f'%(adjrsq_PCoA_dims),fontsize = axFontSize)
axins.set_xticks(tick_labs); axins.set_yticks(tick_labs); 
axins.tick_params(axis='x',labelsize = axFontSize)
axins.tick_params(axis='y',labelsize = axFontSize)
axins.set_xlabel('Obs. discharge', fontsize = labFontSize)
axins.set_ylabel('Mod. discharge', fontsize = labFontSize)

plt.tight_layout()
if save_figs: 
    plt.savefig('Figures/FIG_02{}.jpg'.format(tag), dpi = 800)

#%% FIGURE S1

xlims = np.array([-10,-4])
plt.figure(3,figsize=(3.5,3.5))
ax = plt.subplot(111)
plt.plot(d18o_pre_3,d2h_pre_3,'go',label='Precp. (9/28-10/13)')
plt.plot(d18o_pre_4,d2h_pre_4,'bD',label='Precp. (10/13-11/2)')
plt.plot(-9.0,-63.8,'ys',label='Precp. long term mean')  # [DOI: 10.1002/hyp.11156]
plt.plot(d18o_df,d2h_df,'rx',label='Streamwater samples',zorder=10)
plt.plot(xlims,xlims*8+10,'k:', label='GMWL',zorder=1)
plt.plot(xlims,xlims*7.6 + 6.1,'k-', label='LMWL',zorder=0) # [DOI: 10.1002/hyp.11156]
plt.legend(frameon=False, loc='upper left', fontsize=legFontSize)

plt.ylabel('$\delta$$^2H$ (‰)', fontsize = labFontSize)
plt.xlabel('$\delta$$^{18}O$ (‰)', fontsize = labFontSize)


try: cbar.set_ticks([0,0.5,1,1.5,2], fontsize = axFontSize)
except: cbar.set_ticks([0,0.5,1,1.5,2])
cbar.ax.tick_params(labelsize=axFontSize)

plt.tight_layout()
if save_figs: 
    plt.savefig('Figures/FIG_S1{}.jpg'.format(tag), dpi = 800)


#%% FIGURE S2
plt.figure(4,figsize=(7,3.5))

plt.xticks(np.arange(1,18))
plt.plot(cnt,PCoA.proportion_explained,'rs-',label='PCoA proportion explained\n per coordiante')
plt.plot(cnt,np.cumsum(PCoA.proportion_explained),'bo-',label='PCoA cumulative\nproportion explained')
plt.xlabel('PCoA coordinate count')
plt.legend(frameon=False)
plt.xlim(0.5,17.5); plt.ylim(-.05,1.05); #plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
if save_figs: 
    plt.savefig('Figures/FIG_S2{}.jpg'.format(tag), dpi = 800)

#%% FIGURE S3
plt.figure(5,figsize=(7,3.5))

plt.plot(cnt,rsq_lst,'cs-',label='PCoA (microbial $\\beta$-diversity)')
plt.plot(cnt,adj_rsq_lst,'bo-',label='PCoA (adjusted)')
plt.plot(1,res2.rvalue**2,'go',label='Shannon index (microbial $\\alpha$-diversity)'%res2.rvalue**2)
plt.plot(1,res1.rvalue**2,'gs',label='Richness (microbial $\\alpha$-diversity)')
plt.plot(2,iso_results.rsquared,'o',color = 'darkorchid', label='Water isotopes'%iso_results.rsquared)


iso_pred
plt.xticks(np.arange(1,18))
plt.xlabel('Number of predictors')
plt.ylabel('Proportion of variation in \nstreamflow volume explained ($r^2$)')
plt.legend(frameon=False)
plt.xlim(0.5,17.5); plt.ylim(-.05,1.05);
plt.legend(loc='center')
plt.legend()

plt.tight_layout()
if save_figs: 
    plt.savefig('Figures/FIG_S3{}.jpg'.format(tag), dpi = 800)
