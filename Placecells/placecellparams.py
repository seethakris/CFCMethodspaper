import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sys
from copy import copy
import scipy.stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from statsmodels.stats.multitest import multipletests
import statsmodels.stats.multicomp as mc


# For plotting styles
PlottingFormat_Folder = '/Users/seethakrishnan/Library/CloudStorage/Box-Box/MultiDayData/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf

PvaluesFolder = '/Users/seethakrishnan/Library/CloudStorage/Box-Box/NoReward/Scripts/Figure1/'
sys.path.append(PvaluesFolder)
from Pvalues import GetPValues

class Combinedplots:
    def __init__(self, FolderName, CombinedDataFolder):
        self.CombinedDataFolder = CombinedDataFolder
        self.FolderName = FolderName

        csvfiles_pfs = [f for f in os.listdir(self.CombinedDataFolder) if f.endswith('.csv')]
        self.trackbins = 5
        self.tracklength = 200
        self.numanimals = len(csvfiles_pfs)
        # Combined pf dataframes into one big dataframe
        self.pfparam_combined = self.combineanimaldataframes(csvfiles_pfs)
        self.animals = self.pfparam_combined['animalname'].unique()
        print(csvfiles_pfs)

    def combineanimaldataframes(self, csvfiles, common_flag=False):
        for n, f in enumerate(csvfiles):
            df = pd.read_csv(os.path.join(self.CombinedDataFolder, f), index_col=0)
            if n == 0:
                combined_dataframe = df
            else:
                combined_dataframe = pd.concat((combined_dataframe, df))

            idx = combined_dataframe.index[(combined_dataframe['Width'] > 120)]
            print(idx)
            combined_dataframe = combined_dataframe.drop(idx)
        return combined_dataframe

    def plot_numcells(self, axis, numcells_df, taskstoplot):
        numcells_df = numcells_df[taskstoplot]
        df = numcells_df.melt(var_name='Task', value_name='numcells')
        print(df)
        sns.boxplot(x='Task', y='numcells', data=df, ax=axis, width=0.5)
        for n, row in numcells_df.iterrows():
            axis.plot(row, 'o-', color='k', markerfacecolor='none')
        axis.set_ylabel('Percentage of place cells')
      

    def plot_pfparams(self, ax, tasks_to_plot, columns_to_plot, alltaskpresent=False, commoncellflag=False):
        # Plot a combined historgram and a boxplot of means
        df_plot = self.pfparam_combined[self.pfparam_combined['Task'].isin(tasks_to_plot)]
        for n1, c in enumerate(columns_to_plot):
            # Plot boxplot
            df = df_plot[['animalname', 'Task', c]]
            group = df.groupby(by=['animalname', 'Task']).mean()[c].reset_index()
            group = group.pivot(index='animalname', columns='Task')
            group.columns = group.columns.droplevel()

            x = [0.25, 1.25, 2.25, 3.25]
            for n, row in group.iterrows():
                ax[n1, 0].plot(x, row, 'ko-', markerfacecolor='none', zorder=2, color='lightgrey')

            group = group.melt(value_name=c)
            group = group.dropna()
            comp1 = mc.MultiComparison(group[c], group['Task'])
            tbl, a1, a2 = comp1.allpairtest(scipy.stats.ttest_ind, method= "bonf")
            print(tbl)
            sns.boxplot(x='Task', y=c, data=group, ax=ax[n1, 0], order=tasks_to_plot, showfliers=False)

            ax[n1, 0].set_xlabel('')
            for n2, t in enumerate(tasks_to_plot):
                d = df_plot[df_plot['Task'] == t][c]
                d = d[~np.isnan(d)]
                ax[n1, 1].hist(d, bins=1000, density=True, cumulative=True, label='CDF',
                               histtype='step', linewidth=0.5)
            
    
    def combine_placecells_pertask(self, fig, axis, taskstoplot):
        pc_activity_dict = {keys: np.asarray([]) for keys in taskstoplot}
        perccells_peranimal = {keys: [] for keys in taskstoplot + ['animal']}
        pcsortednum = {keys: [] for keys in taskstoplot}
        for a in self.animals:
            pf_remapping = np.load(
                os.path.join(self.FolderName, a, 'PlaceCells', '%s_pcs_pertask.npy' % a),
                allow_pickle=True).item()
            pfparams = np.load(
                    os.path.join(self.FolderName, a, 'PlaceCells', f'%s_placecell_data.npz' % a), allow_pickle=True)
            perccells_peranimal['animal'].append(a)
            for t in taskstoplot:
                perccells_peranimal[t].append(
                        (np.size(pfparams['numPFs_incells'].item()[t]) / pfparams['numcells']) * 100)
                # print(t, np.sum(pfparams['numPFs_incells'].item()[t]), np.shape(pf_remapping[t]))
                pc_activity_dict[t] = np.vstack((pc_activity_dict[t], pf_remapping[t])) if pc_activity_dict[
                    t].size else pf_remapping[t]
                
        
        for t in taskstoplot:
            print(t, np.shape(pc_activity_dict[t]))
            pcsortednum[t] = np.argsort(np.nanargmax(pc_activity_dict[t], 1))

        self.plot_placecells_pertask(fig, axis, taskstoplot, pc_activity_dict, pcsortednum)
        perccells_peranimal = pd.DataFrame.from_dict(perccells_peranimal)
        perccells_peranimal = perccells_peranimal.set_index('animal')
        return perccells_peranimal

    def combine_placecells_withtask(self, fig, axis, taskstoplot, tasktocompare='Task1', control_flag=False):
        pc_activity_dict = {keys: np.asarray([]) for keys in taskstoplot}
        for a in self.animals:
            if control_flag:
                pf_remapping = np.load(
                    os.path.join(self.FolderName, a, 'PlaceCells', '%s_pcs_sortedby_sametask%s.npy' % (a, tasktocompare)),
                    allow_pickle=True).item()
            else:
                pf_remapping = np.load(
                    os.path.join(self.FolderName, a, 'PlaceCells', '%s_pcs_sortedby%s.npy' % (a, tasktocompare)),
                    allow_pickle=True).item()
            
            for t in taskstoplot:
                pc_activity_dict[t] = np.vstack((pc_activity_dict[t], pf_remapping[t])) if pc_activity_dict[
                    t].size else pf_remapping[t]

        pcsortednum = {keys: [] for keys in taskstoplot}
        pcsorted = np.argsort(np.nanargmax(pc_activity_dict[taskstoplot[0]], 1))
        for t in taskstoplot:
            pcsortednum[t] = pcsorted

        task_data = pc_activity_dict[taskstoplot[0]][pcsorted, :]
        normalise_data = np.nanmax(task_data, 1)[:, np.newaxis]
        self.plot_placecells_pertask(fig, axis, taskstoplot, pc_activity_dict, pcsortednum,
                                          normalise_data=normalise_data)

        return pc_activity_dict
    
    def plot_placecells_pertask(self, fig, axis, taskstoplot, pc_activity, sorted_pcs, controlflag=0, **kwargs):
        for n, taskname in enumerate(taskstoplot):
            task_data = pc_activity[taskname][sorted_pcs[taskname], :]
            if 'normalise_data' in kwargs.keys():
                normalise_data = task_data / kwargs['normalise_data']
            else:
                normalise_data = task_data / np.nanmax(task_data, 1)[:, np.newaxis]
            normalise_data = np.nan_to_num(normalise_data)

            img = axis[n].imshow(normalise_data,
                                 aspect='auto', cmap='jet', interpolation='nearest', vmin=0, vmax=1.0)

            axis[n].set_xticks([0, 20, 39])
            axis[n].set_xticklabels([0, 100, 200])
            axis[n].set_xlim((0, 39))
            if controlflag:
                axis[n].set_title('Cntrl: %s' % taskname)
            else:
                axis[n].set_title('Exp: %s' % taskname)

            pf.set_axes_style(axis[n], numticks=4)
        axis[0].set_xlabel('Track Length (cm)')
        axis[0].set_ylabel('Cell')

    def trackcorrelation(self, pc_activity_dict, basetask='Task1'):
        correlation = {k: [] for k in pc_activity_dict.keys()}
        pcsorted = np.argsort(np.nanargmax(pc_activity_dict[basetask], 1))
        for cell in np.arange(np.size(pc_activity_dict[basetask], 0)):
            x = pc_activity_dict[basetask][cell, :]
            for t in pc_activity_dict.keys():
                y = pc_activity_dict[t][cell, :]
                correlation[t].append(np.corrcoef(x, y)[0, 1])

        # for keys, values in correlation.items():
        #     correlation[keys] = np.asarray(values)[pcsorted]
        return correlation
    
    def get_com_allanimal(self, fig, axis, taskA, taskB, **kwargs):
        csvfiles_pfs = [f for f in os.listdir(self.CombinedDataFolder) if f.endswith('.csv')]
        CombinedDataFolder = self.CombinedDataFolder
        com_all_animal = np.array([])
        count = 0
        for n, f in enumerate(csvfiles_pfs):
            print(f)
            df = pd.read_csv(os.path.join(CombinedDataFolder, f), index_col=0)
            t1 = df[df['Task'] == taskA]
            t2 = df[df['Task'] == taskB]
            combined = pd.merge(t1, t2, how='inner', on=['CellNumber'],
                                suffixes=(f'_%s' % taskA, f'_%s' % taskB))

            if count == 0:
                com_all_animal = np.vstack((combined[f'WeightedCOM_%s' % taskA] * self.trackbins,
                                            combined[f'WeightedCOM_%s' % taskB] * self.trackbins))
            else:
                com_all_animal = np.hstack(
                    (com_all_animal, np.vstack((combined[f'WeightedCOM_%s' % taskA] * self.trackbins,
                                                combined[f'WeightedCOM_%s' % taskB] * self.trackbins))))
            count += 1
        self.plot_com_scatter_heatmap(fig, axis, com_all_animal, taskA, taskB, self.tracklength)
        return com_all_animal
    
    def plot_com_scatter_heatmap(self, fig, axis, combined_dataset, taskA, taskB, tracklength, threshold=25):
        # Scatter plots
        y = combined_dataset[0, :]
        x = combined_dataset[1, :]
        stayed = np.where(np.abs(y - x) < threshold)[0]
        remapped = np.where(np.abs(y - x) >= threshold)[0]

        axis.scatter(y[stayed], x[stayed], color='k', s=5, edgecolor='k', linewidth=0.5)
        axis.scatter(y[remapped], x[remapped], color='grey', s=5, edgecolor='k', linewidth=0.5)
        axis.plot([0, tracklength], [0, tracklength], linewidth=2, color=".3")
        axis.set_xlabel(taskB)
        axis.set_ylabel(taskA)
        axis.set_title('Center of Mass')
        axis.set_ylim((0, tracklength))
        axis.set_xlim((0, tracklength))

class CommonFunctions:
    @staticmethod
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.nanmean(a), scipy.stats.sem(a, nan_policy='omit')
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, m-h, m+h
