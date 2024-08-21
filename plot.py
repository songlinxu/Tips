import time, os, math 
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use("TkAgg")
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd 
import pingouin as pg
import seaborn as sns 
from statsmodels.stats.anova import AnovaRM
from scipy.stats import f_oneway
from scipy import signal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from collections import defaultdict
from EyeTrackingMetrics.eyesmetriccalculator import EyesMetricCalculator
from EyeTrackingMetrics.transition_matrix import *

from math import pi
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
from multiprocessing import Process, Array, Value, Queue
import multiprocessing
import sys

def calculate_anova(data,dependent_variable):
    duration_list = ['short','long']
    condition_list = ['with distraction','no distraction']
    for i,duration in enumerate(duration_list):
        for j,condition in enumerate(condition_list):
            print('For condition: ',duration,condition,', the ANOVA results are below: ')
            data_sub = data[(data['duration_text']==duration)&(data['condition_text']==condition)]
            aov = AnovaRM(data_sub, dependent_variable, 'user_id', within=['threshold_text'])
            results = aov.fit()
            print(results)
            print('The tukey results are below: ')
            tukey_result = pairwise_tukeyhsd(data_sub[dependent_variable], data_sub['threshold_text'])
            print(tukey_result)
            grouped_data = data_sub.groupby('threshold_text')[dependent_variable].agg(['mean', 'std'])
            print('The mean and STD results are below: ')
            print(grouped_data)
    print('For overall conditions, the ANOVA results are below: ')
    aov = AnovaRM(data, dependent_variable, 'user_id', within=['threshold_text','duration_text','condition_text'])
    results = aov.fit()
    print(results)
    print('Overall tukey results are below: ')
    tukey_result = pairwise_tukeyhsd(data[dependent_variable], data['threshold_text'])
    print(tukey_result)
    grouped_data = data.groupby('threshold_text')[dependent_variable].agg(['mean', 'std'])
    print('The overall mean and STD results are below: ')
    print(grouped_data)

def calculate_anova_write(data,dependent_variable,output_name='output'):
    label_dict = {'accuracy_norm':'Normalized Accuracy','missnumber_norm':'Normalized Missed Trial Number','resptime_norm': 'Normalized Response Time', 'gaze_entropy_limit_norm': 'Normalized Gaze Entropy', 'center_distance_norm': 'Normalized Center Distance','accuracy':'Accuracy','missnumber':'Missed Trial Number','resptime': 'Response Time', 'gaze_entropy_limit': 'Gaze Entropy', 'center_distance': 'Center Distance','q1':'Question 1 Score','q2':'Question 2 Score','q3':'Question 3 Score','q4':'Question 4 Score','q5':'Question 5 Score','q6':'Question 6 Score','q1_norm':'Normalized Question 1 Score','q2_norm':'Normalized Question 2 Score','q3_norm':'Normalized Question 3 Score','q4_norm':'Normalized Question 4 Score','q5_norm':'Normalized Question 5 Score','q6_norm':'Normalized Question 6 Score'}

    threshold_rename_dict = {'none':'Silence','static':'Stationary','filtered':'Filter'}
    data['feedback_type'] = data['threshold_text'].map(threshold_rename_dict)

    import pandas as pd 
    from statsmodels.stats.anova import AnovaRM
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    original_stdout = sys.stdout
    output_file_name = output_name+'_'+str(dependent_variable)+'.txt'

    duration_list = ['short','long']
    condition_list = ['with distraction','no distraction']
    with open(output_file_name, 'w') as f:
        sys.stdout = f
        for i,duration in enumerate(duration_list):
            for j,condition in enumerate(condition_list):
                print(f'For variable |{label_dict[dependent_variable]}| in condition: ',duration,condition,', the ANOVA results are below: \n')
                data_sub = data[(data['duration_text']==duration)&(data['condition_text']==condition)]
                aov = AnovaRM(data_sub, dependent_variable, 'user_id', within=['feedback_type'])
                results = aov.fit()
                print(results)
                print('\n')
                print(f'For variable |{label_dict[dependent_variable]}| in condition: ',duration,condition,', the tukey results are below: \n')
                tukey_result = pairwise_tukeyhsd(data_sub[dependent_variable], data_sub['feedback_type'])
                print(tukey_result)
                print('\n')
                grouped_data = data_sub.groupby('feedback_type')[dependent_variable].agg(['mean', 'std'])
                print(f'For variable |{label_dict[dependent_variable]}| in condition: ',duration,condition,', the mean and standard deviation results are below: \n')
                print(grouped_data)
                print('\n')
        print(f'For variable |{label_dict[dependent_variable]}| in all conditions',', the ANOVA results are below: \n')
        aov = AnovaRM(data, dependent_variable, 'user_id', within=['feedback_type','duration_text','condition_text'])
        results = aov.fit()
        print(results)
        print('\n')
        print(f'For variable |{label_dict[dependent_variable]}| in all conditions',', the tukey results are below: \n')
        tukey_result = pairwise_tukeyhsd(data[dependent_variable], data['feedback_type'])
        print(tukey_result)
        print('\n')
        grouped_data = data.groupby('feedback_type')[dependent_variable].agg(['mean', 'std'])
        print(f'For variable |{label_dict[dependent_variable]}| in all conditions',', the overall mean and standard deviation results are below: \n')
        print(grouped_data)
        print('\n')
    



def _smooth_table(input_table,datatype_list):
    output_table = pd.DataFrame()
    new_index = np.arange(input_table.index[0],input_table.index[-1])
    for datatype in datatype_list:
        item = interp1d(input_table.index, input_table[datatype],kind='quadratic')
        output_table[datatype] = item(new_index)
    output_table.index = new_index
    return output_table

def visual_metric_curve(result_path,datatype,unit,dur_interval,smooth = False):
    y_label_dict = {'gaze_entropy_limit_norm':'Normalized Gaze Entropy','gaze_entropy_limit':'Gaze Entropy','center_distance_norm':'Normalized Center Distance','center_distance':'Center Distance'}
    data = pd.read_csv(result_path)
    
    threshold_rename_dict = {'none':'Silence','static':'Stationary','filtered':'Filter'}
    data['threshold_rename'] = data['threshold_text'].map(threshold_rename_dict)


    duration_list = ['short','long']
    condition_list = ['with distraction','no distraction']
    threshold_list = ['Silence','Stationary','Filter']

    fig,axes = plt.subplots(2,2,figsize=(12,6))
    for i,duration in enumerate(duration_list):
        for j,condition in enumerate(condition_list):
            for k,threshold in enumerate(threshold_list):
                print(duration,condition,threshold)
                data_sub = data[(data['duration_text']==duration)&(data['condition_text']==condition)&(data['threshold_rename']==threshold)].copy()
                data_sub[unit+'_second'] = data_sub[unit] * dur_interval
                # Do trim to make sure all conditions have curves within the same data length
                if unit == 'block_id':
                    data_sub = data_sub[(data_sub['block_id']<14)]
                elif unit == 'block_trial_id':
                    if duration == 'short':
                        data_sub = data_sub[(data_sub['block_trial_id']<=15)]
                    else:
                        data_sub = data_sub[(data_sub['block_trial_id']<=120)]
               
                sns.lineplot(data_sub,x=unit+'_second',y=datatype,ax=axes[i][j],label=threshold)
            if unit == 'block_id':
                axes[i][j].set_title('Each trial: '+duration+','+condition)
            elif unit == 'block_trial_id':
                axes[i][j].set_title('Each session: '+duration+','+condition)
            axes[i][j].set_xlabel('Time (second)')
            axes[i][j].set_ylabel(y_label_dict[datatype])
            axes[i][j].spines['top'].set_visible(False)
            axes[i][j].spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()



def calculate_rm_corr(result_path,datatype_x,datatype_y,detrend_flag = False):
    measure_name_dict = {'gaze_entropy_limit_norm': 'Normalized Gaze Entropy', 'resptime_norm': 'Normalized Response Time', 'center_distance_norm': 'Normalized Center Distance', 'gaze_entropy_limit': 'Gaze Entropy', 'resptime': 'Response Time', 'center_distance': 'Center Distance'}
    rename_list = list(measure_name_dict.keys())
    assert datatype_x in rename_list
    assert datatype_y in rename_list
    data = pd.read_csv(result_path)
    data[measure_name_dict[datatype_x]] = data[datatype_x]
    data[measure_name_dict[datatype_y]] = data[datatype_y]
    duration_list = ['short','long']
    condition_list = ['with distraction','no distraction']
    threshold_list = ['none','static','filtered']

    # fig,axes = plt.subplots(1,3)

    for k,threshold in enumerate(threshold_list):
        data_sub = data[(data['threshold_text']==threshold)]
        if detrend_flag == True:
            data_sub[measure_name_dict[datatype_x]] = signal.detrend(data_sub[measure_name_dict[datatype_x]])
            data_sub[measure_name_dict[datatype_y]] = signal.detrend(data_sub[measure_name_dict[datatype_y]])
        try:
            print('seperate: ',threshold)
            r_pval = pg.rm_corr(data=data_sub, x=measure_name_dict[datatype_x], y=measure_name_dict[datatype_y], subject='user_id')
            print(r_pval)
            pg.plot_rm_corr(data=data_sub, x=measure_name_dict[datatype_x], y=measure_name_dict[datatype_y], subject='user_id')

        except: 
            continue
    
    # if detrend_flag == True:
    #     data[datatype_x] = signal.detrend(data[datatype_x])
    #     data[datatype_y] = signal.detrend(data[datatype_y])
    # print('overall: ')
    # r_pval = pg.rm_corr(data=data, x=datatype_x, y=datatype_y, subject='user_id')
    # print('overall: ',f"Repeated Measures Correlation: ",r_pval)
    # g = pg.plot_rm_corr(data=data, x=datatype_x, y=datatype_y, subject='user_id')
    
    plt.show()



def generate_block_result(input_gaze_file,output_file,block_dur):
    # user_id,webstamp,gaze_x_raw,gaze_y_raw,screen_width,screen_height,session,trial,duration_text,condition_text,threshold_text,gaze_x_limit,gaze_y_limit,gaze_x_screen,gaze_y_screen,gaze_x_raw_norm,gaze_y_raw_norm,gaze_x_limit_norm,gaze_y_limit_norm,gaze_x_screen_norm,gaze_y_screen_norm,center_distance

    user_gaze = pd.read_csv(input_gaze_file)
    user_avg_list = []

    user_id_list = list(set(user_gaze['user_id']))

    for i,user_id in enumerate(user_id_list):
        print('user_id: ',user_id)
        for j,duration in enumerate(['short','long']):
            for k,distraction in enumerate(['with distraction','no distraction']):
                for h,threshold in enumerate(['none','static','filtered']):
                    block_trial_id = -1
                    for t,trial_id in enumerate([1,2,3,4,5,6,7,8,9,10]): # 

                        item_gaze = user_gaze[(user_gaze['duration_text']==duration)&(user_gaze['condition_text']==distraction)&(user_gaze['threshold_text']==threshold)&(user_gaze['user_id']==user_id)&(user_gaze['trial']==trial_id)]
                        
                        item_gaze_sorted = item_gaze.sort_values(by='webstamp')

                        starting_time = item_gaze_sorted.iloc[0]['webstamp']
                        block_num = int(np.ceil((item_gaze_sorted.iloc[-1]['webstamp'] - starting_time)/block_dur))
                        for block_id in range(block_num):
                            block_trial_id += 1
                            item_gaze_sorted_piece = item_gaze_sorted[(item_gaze_sorted['webstamp']>(starting_time+block_id*block_dur))&(item_gaze_sorted['webstamp']<(starting_time+(block_id+1)*block_dur))]
                            # print('user_id: ',user_id,' duration: ',duration,' distraction: ',distraction,' threshold: ',threshold,' trial_id: ',trial_id,' data len: ',len(item_gaze_sorted_piece))
                            if len(item_gaze_sorted_piece) == 0: continue

                            center_distance = np.mean(item_gaze_sorted_piece['center_distance'])
                            gaze_entropy_limit = calculate_gaze_entropy(np.array(item_gaze_sorted_piece['gaze_x_limit']),np.array(item_gaze_sorted_piece['gaze_y_limit']),np.array(item_gaze_sorted_piece['screen_width'])[-1],np.array(item_gaze_sorted_piece['screen_height'])[-1])
                            
                            user_avg_list.append([user_id,duration,distraction,threshold,trial_id,block_id,block_trial_id,gaze_entropy_limit,center_distance])

    header = ['user_id','duration_text','condition_text','threshold_text','trial','block_id','block_trial_id','gaze_entropy_limit','center_distance']
    user_avg_arr = np.array(user_avg_list)
    user_avg_table = pd.DataFrame(user_avg_arr,columns=header)
    
    user_avg_table = func_table_norm(user_avg_table,['gaze_entropy_limit','center_distance'])
    user_avg_table.to_csv(output_file,index=False)



def calculate_gaze_entropy(gaze_x_arr,gaze_y_arr,screen_width,screen_height):
    # code adapted from https://github.com/Husseinjd/EyeTrackingMetrics
    TEST_SCREENDIM = [screen_width, screen_height]
    TEST_VERTICES = [[int(screen_width/4),int(screen_height/4)], [int(screen_width/4),int(screen_height/4*3)], [int(screen_width/4*3),int(screen_height/4*3)], [int(screen_width/4*3),int(screen_height/4)]]
    TEST_AOI_DICT = {'aoi_poly1': PolyAOI(TEST_SCREENDIM,TEST_VERTICES)}
    
    GAZE_ARRAY = np.concatenate((gaze_x_arr.reshape((len(gaze_x_arr),1)),gaze_y_arr.reshape((len(gaze_y_arr),1))),axis=1)
    GAZE_ARRAY = np.concatenate((GAZE_ARRAY,np.zeros((len(GAZE_ARRAY),1))),axis=1)

    ec = EyesMetricCalculator(None,GAZE_ARRAY,TEST_SCREENDIM)
        
    gaze_entropy = ec.GEntropy(TEST_AOI_DICT,'stationary').compute()
        
    return gaze_entropy

def func_table_norm(input_table,datatype_list):
    '''Note that this function will change the order of the table data'''
    print('Note that this function will change the order of the table data')
    user_id_list = list(set(input_table['user_id']))
    header = list(input_table.columns.values)
    for datatype in datatype_list:
        header.append(datatype+'_norm')
    for i,user_id in enumerate(user_id_list):
        user_table = input_table[input_table['user_id']==user_id]
        raw_arr = np.array(user_table)
        for j,datatype in enumerate(datatype_list):
            user_table[datatype] = user_table[datatype].astype(float)
            target_arr = np.array(user_table[datatype]).flatten()
            # print(user_id,datatype)
            # print(target_arr)
            arr_std = np.std(target_arr)
            arr_mean = np.mean(target_arr)
            if arr_std == 0:
                print('error! std is zero!!!','!'*20)
                print(user_id,datatype)
                print(target_arr)
                normed_arr = target_arr-arr_mean
            else:
                normed_arr = (target_arr-arr_mean)/arr_std
            normed_arr = normed_arr.reshape((len(normed_arr),1))
            if j==0:
                vertical_arr = normed_arr.copy()
            else:
                vertical_arr = np.concatenate((vertical_arr,normed_arr),axis=1)
        concat_arr = np.concatenate((raw_arr,vertical_arr),axis=1)
        if i==0:
            horiz_arr = concat_arr.copy()
        else:
            horiz_arr = np.concatenate((horiz_arr,concat_arr),axis=0)
    
    table_norm = pd.DataFrame(horiz_arr,columns=header)
    return table_norm


def generate_gaze_norm_file(input_file,output_file):
    user_gaze = pd.read_csv(input_file)
    screen_width_const = np.array(user_gaze['screen_width'])[-1]
    screen_height_const = np.array(user_gaze['screen_height'])[-1]
    gx = np.array(user_gaze['gaze_x_raw']).flatten()
    gy = np.array(user_gaze['gaze_y_raw']).flatten()

    gx_limit = func_limit_arr(gx,screen_width_const,0)
    gy_limit = func_limit_arr(gy,screen_height_const,0)
    gx_limit = gx_limit.reshape((len(gx_limit),1))
    gy_limit = gy_limit.reshape((len(gy_limit),1))

    gx_screen = func_screen_arr(gx_limit,screen_width_const)
    gy_screen = func_screen_arr(gy_limit,screen_height_const)
    gx_screen = gx_screen.reshape((len(gx_screen),1))
    gy_screen = gy_screen.reshape((len(gy_screen),1))

    user_gaze_new = pd.concat([user_gaze,pd.DataFrame(np.concatenate((gx_limit,gy_limit,gx_screen,gy_screen),axis=1),columns=['gaze_x_limit','gaze_y_limit','gaze_x_screen','gaze_y_screen'])],axis=1)

    user_gaze_whole = func_table_norm(user_gaze_new,['gaze_x_raw','gaze_y_raw','gaze_x_limit','gaze_y_limit','gaze_x_screen','gaze_y_screen'])
    user_gaze_whole.to_csv(output_file,index=False)


def generate_avg_results(input_result_file,input_gaze_file,output_file):
    user_result = pd.read_csv(input_result_file)
    user_gaze = pd.read_csv(input_gaze_file)
    user_avg_list = []

    user_id_list = list(set(user_result['user_id']))

    for i,user_id in enumerate(user_id_list):
        for j,duration in enumerate(['short','long']):
            for k,distraction in enumerate(['with distraction','no distraction']):
                for h,threshold in enumerate(['none','static','filtered']):
                    item_data = user_result[(user_result['duration_text']==duration)&(user_result['condition_text']==distraction)&(user_result['threshold_text']==threshold)&(user_result['user_id']==user_id)]
                    item_gaze = user_gaze[(user_gaze['duration_text']==duration)&(user_gaze['condition_text']==distraction)&(user_gaze['threshold_text']==threshold)&(user_gaze['user_id']==user_id)]
                    print(user_id,duration,distraction,threshold,len(item_data),len(item_gaze))
                    acc = np.mean(np.array(item_data['accuracy']))
                    resptime = np.mean(np.array(item_data['resptime'])) 
                    missnumber = 10-len(item_data)
                    screen_width_const = np.array(item_gaze['screen_width'])[-1]
                    screen_height_const = np.array(item_gaze['screen_height'])[-1]
                    gx = np.array(item_gaze['gaze_x_raw']).flatten()
                    gy = np.array(item_gaze['gaze_y_raw']).flatten()
                    gx_raw = gx.copy()
                    gy_raw = gy.copy()
                    gx_raw = gx_raw.reshape((len(gx_raw),1))
                    gy_raw = gy_raw.reshape((len(gy_raw),1))
                    gx_limit = func_limit_arr(gx,screen_width_const,0)
                    gy_limit = func_limit_arr(gy,screen_height_const,0)
                    gx_limit = gx_limit.reshape((len(gx_limit),1))
                    gy_limit = gy_limit.reshape((len(gy_limit),1))
                    gaze_entropy_raw = calculate_gaze_entropy(gx_raw,gy_raw,screen_width_const,screen_height_const)
                    gaze_entropy_limit = calculate_gaze_entropy(gx_limit,gy_limit,screen_width_const,screen_height_const)
                    q1 = np.mean(np.array(item_data['q1']))
                    q2 = np.mean(np.array(item_data['q2']))
                    q3 = np.mean(np.array(item_data['q3']))
                    q4 = np.mean(np.array(item_data['q4']))
                    q5 = np.mean(np.array(item_data['q5']))
                    q6 = np.mean(np.array(item_data['q6']))
                    item_gaze_sorted = item_gaze.sort_values(by='webstamp')
                    session_dur = item_gaze_sorted.iloc[-1]['webstamp'] - item_gaze_sorted.iloc[0]['webstamp']
                    center_distance = np.mean(item_gaze_sorted['center_distance'])
                   
                    user_avg_list.append([user_id,duration,distraction,threshold,acc,resptime,missnumber,gaze_entropy_raw,gaze_entropy_limit,q1,q2,q3,q4,q5,q6,center_distance])


    header = ['user_id','duration_text','condition_text','threshold_text','accuracy','resptime','missnumber','gaze_entropy_raw','gaze_entropy_limit','q1','q2','q3','q4','q5','q6','center_distance']
    user_avg_arr = np.array(user_avg_list)
    user_avg_table = pd.DataFrame(user_avg_arr,columns=header)
    user_avg_table = func_table_norm(user_avg_table,['accuracy','resptime','missnumber','gaze_entropy_raw','gaze_entropy_limit','q1','q2','q3','q4','q5','q6','center_distance'])
    user_avg_table.to_csv(output_file,index=False)

def generate_trial_results(input_result_file,input_gaze_file,output_file):
    user_result = pd.read_csv(input_result_file)
    user_gaze = pd.read_csv(input_gaze_file)
    user_avg_list = []

    user_id_list = list(set(user_result['user_id']))
    trial_id_list = list(set(user_result['trial']))

    for i,user_id in enumerate(user_id_list):
        print(user_id)
        for j,duration in enumerate(['short','long']):
            for k,distraction in enumerate(['with distraction','no distraction']):
                for h,threshold in enumerate(['none','static','filtered']):
                    for t,trial_id in enumerate(trial_id_list):
                        item_data = user_result[(user_result['duration_text']==duration)&(user_result['condition_text']==distraction)&(user_result['threshold_text']==threshold)&(user_result['user_id']==user_id)&(user_result['trial']==trial_id)]
                        item_gaze = user_gaze[(user_gaze['duration_text']==duration)&(user_gaze['condition_text']==distraction)&(user_gaze['threshold_text']==threshold)&(user_gaze['user_id']==user_id)&(user_gaze['trial']==trial_id)]
                        if len(item_data) == 0: continue 
                        # print(user_id,duration,distraction,threshold,len(item_data),len(item_gaze))
                        acc = np.mean(np.array(item_data['accuracy']))
                        resptime = np.mean(np.array(item_data['resptime'])) 
                        screen_width_const = np.array(item_gaze['screen_width'])[-1]
                        screen_height_const = np.array(item_gaze['screen_height'])[-1]
                        gx = np.array(item_gaze['gaze_x_raw']).flatten()
                        gy = np.array(item_gaze['gaze_y_raw']).flatten()
                        gx_raw = gx.copy()
                        gy_raw = gy.copy()
                        gx_raw = gx_raw.reshape((len(gx_raw),1))
                        gy_raw = gy_raw.reshape((len(gy_raw),1))
                        gx_limit = func_limit_arr(gx,screen_width_const,0)
                        gy_limit = func_limit_arr(gy,screen_height_const,0)
                        gx_limit = gx_limit.reshape((len(gx_limit),1))
                        gy_limit = gy_limit.reshape((len(gy_limit),1))
                        gaze_entropy_raw = calculate_gaze_entropy(gx_raw,gy_raw,screen_width_const,screen_height_const)
                        gaze_entropy_limit = calculate_gaze_entropy(gx_limit,gy_limit,screen_width_const,screen_height_const)
                        q1 = np.mean(np.array(item_data['q1']))
                        q2 = np.mean(np.array(item_data['q2']))
                        q3 = np.mean(np.array(item_data['q3']))
                        q4 = np.mean(np.array(item_data['q4']))
                        q5 = np.mean(np.array(item_data['q5']))
                        q6 = np.mean(np.array(item_data['q6']))
                        item_gaze_sorted = item_gaze.sort_values(by='webstamp')
                        trial_dur = item_gaze_sorted.iloc[-1]['webstamp'] - item_gaze_sorted.iloc[0]['webstamp']
                        center_distance = np.mean(item_gaze_sorted['center_distance'])
                       
                        user_avg_list.append([user_id,duration,distraction,threshold,trial_id,acc,resptime,gaze_entropy_raw,gaze_entropy_limit,q1,q2,q3,q4,q5,q6,center_distance])


    header = ['user_id','duration_text','condition_text','threshold_text','trial','accuracy','resptime','gaze_entropy_raw','gaze_entropy_limit','q1','q2','q3','q4','q5','q6','center_distance']
    user_avg_arr = np.array(user_avg_list)
    user_avg_table = pd.DataFrame(user_avg_arr,columns=header)
    user_avg_table = func_table_norm(user_avg_table,['resptime','gaze_entropy_raw','gaze_entropy_limit','q1','q2','q3','q4','q5','q6','center_distance'])
    user_avg_table.to_csv(output_file,index=False)

def generate_gaze_extension(input_result_file,output_file):
    ext_header = ['center_distance']
    dataset = pd.read_csv(input_result_file)

    def calculate_center_distance(col1, col2):
        return math.sqrt((col1-0.5)* (col1-0.5) + (col2-0.5)*(col2-0.5))

    # Apply the function and assign the result to a new column 'SumColumn'
    dataset['center_distance'] = dataset.apply(lambda row: calculate_center_distance(row['gaze_x_screen'], row['gaze_y_screen']), axis=1)

    # center_distance_arr
    dataset.to_csv(output_file,index=False)






def visual_heatmap_subjective_questionnaire(result_file):
    data = pd.read_csv(result_file)
    threshold_rename_dict = {'none':'Silence','static':'Stationary','filtered':'Filter'}
    data['threshold_rename'] = data['threshold_text'].map(threshold_rename_dict)
    duration_list = ['short','long']
    condition_list = ['with distraction','no distraction']
    threshold_list = ['Silence','Stationary','Filter']
    question_list = [q_text for q_text in ['q1','q2','q3','q4','q5','q6']]
    # question_list = [q_text+'_norm' for q_text in ['q1','q2','q3','q4','q5','q6']]
    fig,axes=plt.subplots(2,2)
    fig2,axes2=plt.subplots(2,2)
    custom_xticklabels = question_list
    custom_yticklabels = threshold_list
    matrix_list_all = []
    
    for d,duration in enumerate(duration_list):
        for c,condition in enumerate(condition_list):
            data_part = np.empty((0,4))
            matrix_list = []
            for t,threshold in enumerate(threshold_list):
                piece_list = []
                data_sub = data[(data['duration_text']==duration)&(data['condition_text']==condition)&(data['threshold_rename']==threshold)]
                for q,question in enumerate(question_list):
                    piece_list.append(np.mean(data_sub[question]))
                    datalen = len(data_sub)
                    # print('datalen: ',datalen)
                    user_id_arr = np.array(data_sub['user_id']).reshape((datalen,1))
                    threshold_arr = np.array([threshold for tid in range(datalen)]).reshape((datalen,1))
                    score_arr = np.array(data_sub[question]).reshape((datalen,1))
                    question_id_arr = np.array([question for tid in range(datalen)]).reshape((datalen,1))
                    data_piece = np.concatenate((user_id_arr,threshold_arr,score_arr,question_id_arr),axis=1)
                    data_part = np.concatenate((data_part,data_piece),axis=0)


                matrix_list.append(piece_list)
                matrix_list_all.append(piece_list)

            sns.heatmap(matrix_list, cmap='viridis', annot=True, fmt=".2f", linewidths=.5, ax=axes[d][c], xticklabels=custom_xticklabels, yticklabels=custom_yticklabels)
            
            data_part = pd.DataFrame(data_part,columns=['user_id','threshold_rename','question_score','question_id'])
            data_part['question_score'] = data_part['question_score'].astype(float)
            sns.barplot(data_part, x='question_id', y='question_score', hue='threshold_rename', ax=axes2[d][c])
            axes2[d][c].set_xlabel('Question ID')
            axes2[d][c].set_ylabel('Question Score')
            axes2[d][c].legend(title='') 
    # sns.heatmap(matrix_list, annot=True, cmap='Blues', fmt=".2f", vmax=.8, center=0,square=True, linewidths=3, annot_kws={"fontsize":12},cbar_kws={"shrink": .5, 'location': 'right', 'orientation': 'vertical', 'pad': 0})
    
    # sns.heatmap(matrix_list_all, cmap='viridis', annot=True, fmt=".2f", linewidths=.5, ax=axes[d][c], xticklabels=custom_xticklabels, yticklabels=custom_yticklabels)
    plt.subplots_adjust(bottom=0.1)
    plt.show()

def visual_heatmap_individual(table_file,user_id_list,datatype):
    title_dict = {'gaze_entropy_limit_norm':'Normalized Gaze Entropy Heatmap','gaze_entropy_limit':'Gaze Entropy Heatmap','center_distance_norm':'Normalized Center Distance Heatmap','center_distance':'Center Distance Heatmap'}
    user_table_all = pd.read_csv(table_file)
    user_table_select = pd.DataFrame(columns=user_table_all.columns.values)
    if user_id_list != None:
        for i in user_id_list:
            user_item = user_table_all[(user_table_all['user_id']==i)]
            user_table_select = pd.concat([user_table_select,user_item],axis=0)
    else:
        user_table_select = user_table_all
    fig, axes = plt.subplots(1,1,figsize = (5,7))
    matrix_data = np.empty((0,21))
    for i,duration in enumerate(['short','long']):
        for j,distraction in enumerate(['with distraction', 'no distraction']):
            line_data = np.empty((3,0))
            for k,threshold in enumerate(['none','static','filtered']):
                result_data_select = user_table_select[(user_table_select['duration_text']==duration)&(user_table_select['condition_text']==distraction)&(user_table_select['threshold_text']==threshold)]
                line_data = np.concatenate((line_data,np.array(result_data_select[datatype]).reshape((3,7))),axis=1)
            matrix_data = np.concatenate((matrix_data,line_data),axis=0)
                    
    print(matrix_data.shape)
    sns.heatmap(data = np.array(matrix_data).T,xticklabels=False, yticklabels=False,ax=axes)
    if datatype in list(title_dict.keys()):
        axes.set_title(title_dict[datatype])
    else:
        axes.set_title(datatype)
   
    plt.show()

def visual_radar_score(data_csv,duration,distraction,norm = ''):
    data = pd.read_csv(data_csv)
    threshold_rename_dict = {'none':'Silence','static':'Stationary','filtered':'Filter'}
    data['threshold_rename'] = data['threshold_text'].map(threshold_rename_dict)

    if duration != None:
        data = data[(data['duration_text']==duration)]
    if distraction != None:
        data = data[(data['condition_text']==distraction)]
    table_list = []
    for i,threshold in enumerate(['Silence','Stationary','Filter']):
        data_item = data[data['threshold_rename']==threshold]
        data_len = len(data_item)
        data_arr = np.concatenate((np.array(data_item['q1'+norm]).reshape((data_len,1)),np.array(data_item['q2'+norm]).reshape((data_len,1)),np.array(data_item['q3'+norm]).reshape((data_len,1)),np.array(data_item['q4'+norm]).reshape((data_len,1)),np.array(data_item['q5'+norm]).reshape((data_len,1)),np.array(data_item['q6'+norm]).reshape((data_len,1))),axis=1)
        table_list.append([threshold]+list(np.mean(data_arr,axis=0)))

    if norm == '':
        table = pd.DataFrame(np.array(table_list),columns=['threshold_rename','Q1','Q2','Q3','Q4','Q5','Q6'])
    elif norm == '_norm':
        table = pd.DataFrame(np.array(table_list),columns=['threshold_rename','Normalized Q1','Normalized Q2','Normalized Q3','Normalized Q4','Normalized Q5','Normalized Q6'])
    print(table)
    color_dict = {'Silence':'orange','Stationary':'g','Filter':'blue'}
    visual_radar(table,hue='threshold_rename',color_dict=color_dict,step=3)


def visual_radar(data,hue,color_dict,step = 3):
    # code adapted from https://python-graph-gallery.com/391-radar-chart-with-several-individuals/
    group_list = list(set(data[hue]))
    categories=list(data)[1:]
    N = len(categories)

    data_value = np.float_(np.array(data)[:,1:])
    
    max_value = np.max(data_value)
    min_value = np.min(data_value)
    print('max: ', max_value, ' min: ', min_value)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
 
    # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax = plt.subplot(111,polar=True)
    ax.set_facecolor('#f0f2f5')
 
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
 
    plt.xticks(angles[:-1], categories)
 
    ax.set_rlabel_position(0)

    max_limit = max_value*1.1 if max_value > 0 else max_value*0.9
    min_limit = min_value*0.9 if min_value > 0 else min_value*1.1

    interval = (max_limit - min_limit)/step
    ytick_list = [round(min_limit + interval * (k+1),3) for k in range(step)]
    ytick_str = [str(ki) for ki in ytick_list]
    # ytick_str = [str((min_limit+interval * (k+1))) for k in range(step)]
    print(ytick_list)
    print(ytick_str)

    ax.grid(True, linestyle='--', linewidth=1)
    ax.spines['polar'].set_visible(False)

    plt.yticks(ytick_list, ytick_str, color="grey", size=7)
    # plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
    plt.ylim(min_limit,max_limit)

    Question_Label_Dict = {'q1': 'Q1', 'q2': 'Q2', 'q3': 'Q3', 'q4': 'Q4', 'q5': 'Q5', 'q6': 'Q6', 'q1_norm': 'Normalized Q1', 'q2_norm': 'Normalized Q2', 'q3_norm': 'Normalized Q3', 'q4_norm': 'Normalized Q4', 'q5_norm': 'Normalized Q5', 'q6_norm': 'Normalized Q6'}
    for i in range(len(group_list)):
        df_item = data[data[hue]==group_list[i]]
        values=list(np.float_(np.array(df_item)[0][1:]))
        values += values[:1]

        X_Y_Spline = make_interp_spline(angles[:-1], values[:-1])
        X_ = np.linspace(min(angles), max(angles), 500)
        Y_ = X_Y_Spline(X_)

        X_ = list(X_)
        Y_ = list(Y_)

        X_ += angles[:1]
        Y_ += values[:1]

        ax.plot(angles, values, linewidth=1, linestyle='solid', marker='o', markersize=10, alpha=0.8, label=group_list[i], c=color_dict[group_list[i]])
        # ax.fill(angles, values, color_dict[group_list[i]], alpha=0.1)
 
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.show()



def visual_gaze(gaze_all_file,user_id,tail='raw'):
    # warning: this code function is only suitable for one user, not multiple users together. 
    gaze_data_all = pd.read_csv(gaze_all_file)
    color_dict = {'none':'b','filtered':'g','static':'r'}
    user_table_select = gaze_data_all[gaze_data_all['user_id']==user_id]
    fig, axes = plt.subplots(3,4,figsize = (18,10))
    for i,duration in enumerate(['short','long']):
        for j,distraction in enumerate(['with distraction', 'no distraction']):
            for k,feedback in enumerate(['none','static','filtered']):
                gaze_data_select = user_table_select[(user_table_select['duration_text']==duration)&(user_table_select['condition_text']==distraction)&(user_table_select['threshold_text']==feedback)]

                x_data = gaze_data_select['gaze_x_'+tail]
                y_data = gaze_data_select['gaze_y_'+tail]
                
                axes[k][i*2+j].scatter(x_data,y_data,s=4,c=color_dict[feedback])
                axes[k][i*2+j].set(facecolor = "black")
                axes[k][i*2+j].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

    plt.subplots_adjust(hspace = 0.03,wspace = 0.03)
    
    plt.show()




def plot_bar(data_csv,datatype = 'resptime'):
    user_table_select = pd.read_csv(data_csv)
    sns.barplot(user_table_select,x='threshold_text',y=datatype)
    sns.despine()
    plt.show()

def plot_box(data_csv,datatype = 'resptime'):
    label_dict = {'resptime_norm': 'Normalized Response Time', 'gaze_entropy_limit_norm': 'Normalized Gaze Entropy', 'center_distance_norm': 'Normalized Center Distance','accuracy':'Accuracy','missnumber':'Missed Trial Number','resptime': 'Response Time', 'gaze_entropy_limit': 'Gaze Entropy', 'center_distance': 'Center Distance'}
    threshold_rename_dict = {'none':'Silence','static':'Stationary','filtered':'Filter'}
    user_table_select = pd.read_csv(data_csv)
    user_table_select['threshold_rename'] = user_table_select['threshold_text'].map(threshold_rename_dict)

    fig, axes = plt.subplots(2,2,figsize = (6,6))
    for i,duration in enumerate(['short','long']):
        for j,distraction in enumerate(['with distraction', 'no distraction']):
            user_table_item = user_table_select[(user_table_select['duration_text']==duration)&(user_table_select['condition_text']==distraction)].copy()
            sns.stripplot(data=user_table_item, x="threshold_rename", y=datatype,dodge=False, alpha=.5, legend=False,ax=axes[i][j],palette=['b', 'g', 'y'],jitter=False)
            sns.boxplot(user_table_item,x='threshold_rename',y=datatype,ax=axes[i][j],palette=['lightblue', 'lightgreen', 'lightyellow'],showfliers=False,width=0.4,linewidth=1.5)
            sns.lineplot(data=user_table_item, x="threshold_rename", y=datatype, units="user_id",color=".7", estimator=None,ax=axes[i][j],linestyle='dashed',linewidth=0.5)

            axes[i][j].set_xlabel(duration+','+distraction)
            if datatype in list(label_dict.keys()):
                axes[i][j].set_ylabel(label_dict[datatype])
            else:
                axes[i][j].set_ylabel(datatype)
            # axes[i][j].set_xticklabels(['Silence','Stationary','Filter'])
    sns.despine()
    plt.tight_layout()
    # plt.subplots_adjust(hspace = 0.5,wspace = 0.5,bottom=0)
    plt.show()



def func_limit_arr(input_arr, max_value, min_value):
    new_arr = []
    for arr_item in input_arr:
        if arr_item > max_value:
            new_arr.append(max_value)
        elif arr_item < min_value:
            new_arr.append(min_value)
        else:
            new_arr.append(arr_item)
    return np.array(new_arr)


def func_screen_arr(input_arr, screen_value):
    new_arr = []
    for arr_item in input_arr:
        new_arr.append(arr_item/screen_value)
    return np.array(new_arr)
