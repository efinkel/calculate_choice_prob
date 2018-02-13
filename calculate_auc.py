"""
module containing functions for calculating the choice probability of recorded neurons
1/19/2018
Eric Finkel
"""
import time

import scipy as sp
import scipy.io
import scipy.stats
import os
import numpy as np
import pandas as pd
import glob
import csv
import random as rand
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import multiprocessing as mp
from tqdm import tnrange
from itertools import repeat
from functools import partial
import tqdm
from tqdm import tnrange
import parmap


def load_data():
	"""
	function that loads, cleans, and performs some initial processing on log_df table that contains data from entire dataset.
	log_df table was generated from Intan recording data that was originally preprocessed in matlab using
	cat_session.mat function
	"""
	
	log_df = pd.read_hdf('C:/Users/PC/Documents/GitHub/Crossmodal-project/Data/log_df.h5', 'table')
	log_df['stim_onset'] = log_df['stim_onset'].fillna(0)
	log_df['spike_times(stim_aligned)'] = log_df['spike_times'] - log_df['stim_onset']
	log_df = log_df[~log_df['trial_type'].str.contains('NoStim')]
	licks = pd.concat([log_df['licks_right'] - log_df['stim_onset'] , log_df['licks_left']-log_df['stim_onset']], axis=1)
	licks = licks.applymap(lambda y: y[[0.1<y]] if len(y) > 0 else y)
	licks = licks.applymap(lambda y: y[[3>=y]] if len(y) > 0 else y)
	first_licks = licks.applymap(lambda y: min(y) if len(y) > 0 else np.nan)
	last_licks = licks.applymap(lambda y: max(y) if len(y) > 0 else np.nan)

	log_df['first_lick'] = first_licks.min(axis=1)
	log_df['last_lick'] = last_licks.max(axis=1)

	log_df['spike_times(stim_aligned)'] = log_df['spike_times(stim_aligned)'].apply(lambda x: np.concatenate(x) 
																					   if len(list(x)) > 0 else x)
																					   
	log_df = log_df.sort_values(['mouse_name', 'date', 'cluster_name', 'first_lick'], ascending = [1,1,1,1])
	log_df['identified'] = 'unidentified'
	log_df = log_df.reset_index(drop=True)

	log_df['correct'] = 0

	# fwd contingencies - correct responses
	log_df.loc[~(log_df['mouse_name'].isin(['EF0091', 'EF0099', 'EF0101', 'EF0102'])) & (log_df['block_type'] == 'Whisker') &
		   (log_df['trial_type'].str.contains('Som')) & (log_df['response'] == 1), 'correct'] = 1
	log_df.loc[~(log_df['mouse_name'].isin(['EF0091', 'EF0099', 'EF0101', 'EF0102'])) & (log_df['block_type'] == 'Visual') &
		   (log_df['trial_type'].str.contains('Vis')) & (log_df['response'] == 2), 'correct'] = 1

	#rev contingencies - correct responses
	log_df.loc[(log_df['mouse_name'].isin(['EF0091', 'EF0099', 'EF0101', 'EF0102'])) & (log_df['block_type'] == 'Whisker') &
		   (log_df['trial_type'].str.contains('Som')) & (log_df['response'] == 2), 'correct'] = 1
	log_df.loc[(log_df['mouse_name'].isin(['EF0091', 'EF0099', 'EF0101', 'EF0102'])) & (log_df['block_type'] == 'Visual') &
		   (log_df['trial_type'].str.contains('Vis')) & (log_df['response'] == 1), 'correct'] = 1

	#correct rejections
	log_df.loc[(log_df['block_type'] == 'Whisker') & (log_df['trial_type'].str.contains('Vis'))
			   & (log_df['response'] == 0), 'correct'] = 1
	log_df.loc[(log_df['block_type'] == 'Visual') & (log_df['trial_type'].str.contains('Som'))
			   & (log_df['response'] == 0), 'correct'] = 1
	
	
	log_df['uni_id'] = (log_df['mouse_name'].apply(lambda x: x[-3:]) + log_df['date']+
							 log_df['cluster_name'].apply(lambda x: x[2] + x[-2:]))
	unit_key_df = log_df[['uni_id', 'mouse_name', 'date', 'cluster_name']].drop_duplicates().reset_index(drop = True)
	return log_df, unit_key_df
def filt_motion_trials(log_df, exclude_fn):
	"""
	takes exclude_fn df and uses contents to filter out rows with high motion artifact
	in main log_df file
	"""
	
	mat = sp.io.loadmat(exclude_fn)
	log = mat['trialsToExclude']
	indv_log_df = pd.DataFrame(log, columns = ['mouse_name', 'date', 'trial_num'])
	
	for col in [0,1,2,2]:
		indv_log_df.ix[:,col] = indv_log_df.ix[:,col].str[0]
	
	log_df.reset_index(inplace = True)
	
	rows_to_exclude = pd.merge(log_df, indv_log_df, how='inner')
	inds_to_exclude = rows_to_exclude['index'].as_matrix()
	log_df = log_df.drop(inds_to_exclude).reset_index(drop=True)
	
	return log_df

def unit_row_list(log_df):
	"""cuts up main log_df file into a list of dfs that each correspond to one
	   unit. This should increase performance of trial_auc function ~10s/unit
	"""
	
	log_byID = log_df.groupby('uni_id')
	unique_ids = log_byID.groups.keys()
	log_unit_list = [log_byID.get_group(key) for key in unique_ids]
	return log_unit_list
	
def get_spike_counts(unit_rows, trial_type, comparison):
	
	if comparison == 'Lick_no_lick':
		pos_rows = unit_rows.loc[(unit_rows['trial_type'] == trial_type) & (unit_rows['response'] != 0),:].copy()
		neg_rows = unit_rows.loc[(unit_rows['trial_type'] == trial_type) & (unit_rows['response'] == 0),:].copy()
	elif comparison == 'touch_hit_miss':
		pos_rows = unit_rows.loc[(unit_rows['trial_type'] == trial_type) & (unit_rows['response'] != 0) & (unit_rows['block_type'] == 'Whisker'):].copy()
		neg_rows = unit_rows.loc[(unit_rows['trial_type'] == trial_type) & (unit_rows['response'] == 0) & (unit_rows['block_type'] == 'Whisker'):].copy()
	elif comparison == 'touch_hit_cr':
		pos_rows = unit_rows.loc[(unit_rows['trial_type'] == trial_type) & (unit_rows['response'] != 0) & (unit_rows['block_type'] == 'Whisker'):].copy()
		neg_rows = unit_rows.loc[(unit_rows['trial_type'] == trial_type) & (unit_rows['response'] == 0) & (unit_rows['block_type'] == 'Visual'):].copy()
	elif comparison == 'touch_miss_cr':
		pos_rows = unit_rows.loc[(unit_rows['trial_type'] == trial_type) & (unit_rows['response'] == 0) & (unit_rows['block_type'] == 'Whisker'):].copy()
		neg_rows = unit_rows.loc[(unit_rows['trial_type'] == trial_type) & (unit_rows['response'] == 0) & (unit_rows['block_type'] == 'Visual'):].copy()
	elif comparison == 'visual_hit_miss':
		pos_rows = unit_rows.loc[(unit_rows['trial_type'] == trial_type) & (unit_rows['response'] != 0) & (unit_rows['block_type'] == 'Visual'):].copy()
		neg_rows = unit_rows.loc[(unit_rows['trial_type'] == trial_type) & (unit_rows['response'] == 0) & (unit_rows['block_type'] == 'Visual'):].copy()
	elif 'visual_hit_cr':
		pos_rows = unit_rows.loc[(unit_rows['trial_type'] == trial_type) & (unit_rows['response'] != 0) & (unit_rows['block_type'] == 'Visual'):].copy()
		neg_rows = unit_rows.loc[(unit_rows['trial_type'] == trial_type) & (unit_rows['response'] == 0) & (unit_rows['block_type'] == 'Whisker'):].copy()
	elif comparison == 'visual_miss_cr':
		pos_rows = unit_rows.loc[(unit_rows['trial_type'] == trial_type) & (unit_rows['response'] == 0) & (unit_rows['block_type'] == 'Visual'):].copy()
		neg_rows = unit_rows.loc[(unit_rows['trial_type'] == trial_type) & (unit_rows['response'] == 0) & (unit_rows['block_type'] == 'Whisker'):].copy()
	elif comparison == 'lick_right':
		pos_rows = unit_rows.loc[(unit_rows['trial_type'] == trial_type) & (unit_rows['response'] == 1):].copy()
		neg_rows = unit_rows.loc[(unit_rows['trial_type'] == trial_type) & (unit_rows['response'] == 0):].copy()
	elif  comparison == 'lick_left':
		pos_rows = unit_rows.loc[(unit_rows['trial_type'] == trial_type) & (unit_rows['response'] == 2):].copy()
		neg_rows = unit_rows.loc[(unit_rows['trial_type'] == trial_type) & (unit_rows['response'] == 0):].copy()
	else:
		raise ValueError("Invalid comparison passed. Pass either: 'touch_hit_miss', 'touch_hit_cr', 'touch_miss_cr', 'visual_hit_miss', 'visual_hit_cr', 'visual_hit_cr', 'visual_miss_cr'")
	
	if ~pos_rows.empty: pos_rows['labels'] = 1
	if ~neg_rows.empty: neg_rows['labels'] = 0
	
	comparison_rows = pd.concat([pos_rows, neg_rows])
	edges = np.arange(-1,3, 0.025)

	spike_counts_df = comparison_rows[['mouse_name', 'date', 'cluster_name',
									   'trial_num', 'spike_times(stim_aligned)', 'labels']].copy().reset_index(drop=True)
	spike_counts_df['spike_counts(stim_aligned)'] = spike_counts_df['spike_times(stim_aligned)'].apply(lambda y: np.histogram(y, edges)[0])
	
	y = spike_counts_df['labels'].as_matrix()
	if spike_counts_df.shape[0] == 0:
		spike_counts_2d = []
	else:
		spike_counts_2d = np.stack(spike_counts_df['spike_counts(stim_aligned)'].as_matrix())

	return spike_counts_2d, y


def trial_auc(binned_FR, y_values):
	"""
	function that calculates the binned auc values comparing two user defined trial types.
	outputs three numpy array containing binned auc values, upper confidence interval, lower confidence interval
	"""
	
	if len(np.unique(y_values)) < 2:
		auc_scores = np.array([np.nan]*159, ndmin=2)
		unit_conf_upper = np.array([np.nan]*159, ndmin=2)
		unit_conf_lower = np.array([np.nan]*159, ndmin=2)
		return auc_scores, unit_conf_lower, unit_conf_upper

	auc_scores = np.zeros([1,binned_FR.shape[1]])
	unit_conf_upper = np.zeros([1,binned_FR.shape[1]])
	unit_conf_lower = np.zeros([1,binned_FR.shape[1]])
	for bin in range(binned_FR.shape[1]):
		auc_scores[0][bin] = roc_auc_score(y_values,binned_FR[:, bin])
		
		rng_seed = 42  # control reproducibility
		rng = np.random.RandomState(rng_seed)
		indices = rng.randint(0, len(y_values) - 1, [1000, len(y_values)])
		
		bootstrapped_scores = [roc_auc_score(y_values[indices[boot_num,:]], binned_FR[indices[boot_num,:], bin]) for 
								boot_num in range(1000) if len(np.unique(y_values[indices[boot_num,:]])) == 2]
		sorted_scores = np.array(bootstrapped_scores)
		sorted_scores.sort()
		unit_conf_lower[0][bin] = sorted_scores[int(0.025 * len(sorted_scores))]
		unit_conf_upper[0][bin] = sorted_scores[int(0.975 * len(sorted_scores))]
	return auc_scores, unit_conf_lower, unit_conf_upper

def package_auc(unit_key_df, trial_type, comparison, auc_scores, conf_upper, conf_lower):

    """
    function that turns auc scores and confidence interval arrrays into a data frame containing metadata
    describing the details of the auc calculation
    """
    unit_key_df['tt_comp'] = trial_type
    unit_key_df['comparison'] = comparison

    auc_labels = ['auc_score'+str(x) for x in range(len(auc_scores[0]))]
    conf_upper_labels = ['conf_upper'+str(x) for x in range(len(auc_scores[0]))]
    conf_lower_labels = ['conf_lower'+str(x) for x in range(len(auc_scores[0]))]

    auc_scores_df = pd.DataFrame(auc_scores, columns = auc_labels)
    conf_up_df = pd.DataFrame(conf_upper, columns = conf_upper_labels)
    conf_low_df = pd.DataFrame(conf_lower, columns = conf_lower_labels)
    
            
    auc_df = pd.concat([unit_key_df, auc_scores_df, conf_up_df, conf_low_df], axis = 1)
    return auc_df
	
def main(trial_type = 'Stim_Som_NoCue', comparison = 'lick_right'):

	log_df, unit_key_df = load_data()
	unique_ids = unit_key_df['uni_id'].as_matrix()
	unit_list = unit_row_list(log_df)
	log_df = filt_motion_trials(log_df, 'trialsToExclude')
	start_time = time.time()
	
	binned_FRs, y_values = zip(*[get_spike_counts(unit_list[unit], trial_type, comparison) 
								for unit in range(len(unit_list))])
	
	with mp.Pool(7) as pool:
		aucs_CI = pool.starmap(trial_auc, zip(binned_FRs, y_values))
	#auc_CI = parmap.starmap(trial_auc, unit_list[0:5], trial_type)
	# aucs_CI = [trial_auc(unit_list[unit], trial_type) for unit in range(len(unit_list[0:200]))]
	aucs_CI = np.squeeze(np.array(aucs_CI))

	auc_scores = aucs_CI[:,0]
	conf_upper = aucs_CI[:,1]
	conf_lower = aucs_CI[:,2]
	df = package_auc(unit_key_df, trial_type, comparison, auc_scores, conf_upper, conf_lower)
	df.to_hdf('touch_lick_right_auc.h5', 'table')
	print(df)
	print("--- %s seconds ---" % (time.time() - start_time))
	
if __name__ == '__main__':
	main()
		