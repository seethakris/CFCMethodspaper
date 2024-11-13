import scipy.io
import scipy.signal
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from ast import literal_eval

class GetData(object):
    def __init__(self, FolderName):
        self.FolderName = FolderName
        self.csvfiles = [f for f in os.listdir(self.FolderName) if f.endswith('.csv')]
        print(self.csvfiles)
    
    def get_averagelaptime(self, plot=False):
        combined_df = pd.DataFrame()

        for i in self.csvfiles:
            print('Parsing...', i)
            df = pd.read_csv(os.path.join(self.FolderName, i))
            df = df.iloc[: , 1:]

            df['Behavior'] = df['Behavior'].apply(literal_eval)
            df['Lap'] = df['Lap'].apply(literal_eval)
            
            df['Laptime_bylap'] = np.nan
            df['Laptime_bylap'] = df['Laptime_bylap'].astype('object')
            df['Avglaptime'] = np.nan
            df['Firstlap_time'] = np.nan
            df['Numlaps'] = np.nan
            df['TotalTime'] = np.nan
            df['Freezing_bylap'] = np.nan
            df['Freezing_bylap'] = df['Freezing_bylap'].astype('object')
            df['Freezingpercent_bylap'] = np.nan
            df['Freezingpercent_bylap'] = df['Freezingpercent_bylap'].astype('object')
            df['Firstlapfreezing'] = np.nan
            df['Averagefreezing'] = np.nan
            df['Freezingepoch_bylap'] = np.nan
            df['Freezingepoch_bylap'] = df['Freezingepoch_bylap'].astype('object')

            print(len(df))
            for ind, row in df.iterrows():
                if plot:
                    fs, ax = plt.subplots(1, figsize=(6, 1))
                    ax.plot(row['Behavior'])
                    ax.plot(row['Lap'])
                    ax.set_title('Animal: %s, Shocktype: %s' %(row['Animal'], row['Shocktype']))
                
                lapdata = np.asarray(row['Lap'])
                behdata = np.asarray(row['Behavior'])

                laps = np.unique(lapdata)

                avg_laplength = []
                for l in laps:
                    if l>0 and l<laps[-1]:
                        temp = np.size(np.where(lapdata==l)[0])/1000
                        if temp>1:
                            avg_laplength.append(temp)

                #Get frames with freezing
                freezetime, freezepercent, freezeepoch = self.get_freezing(behdata, lapdata)
                
                df.at[ind, 'Laptime_bylap'] = avg_laplength
                df.loc[ind, 'Firstlap_time'] = np.mean(avg_laplength[0:2])
                df.loc[ind, 'Numlaps'] = laps[-2]
                df.loc[ind, 'Avglaptime'] = np.mean(avg_laplength)
                df.loc[ind, 'TotalTime'] = np.size(lapdata[np.where(lapdata==1)[0][0]:np.where(lapdata==laps[-1])[0][0]])/1000

                df.at[ind, 'Freezing_bylap'] = freezetime
                df.at[ind, 'Freezingpercent_bylap'] = freezepercent
                df.loc[ind, 'Firstlapfreezing'] = np.nanmean(freezepercent[0:2])
                df.loc[ind, 'Averagefreezing'] = np.nanmean(freezepercent)
                df.at[ind, 'Freezingepoch_bylap'] = freezeepoch

            combined_df = pd.concat((combined_df, df))
        combined_df = combined_df.reset_index()
        return combined_df

    
    def get_freezing(self, beh, lap):
        fs, ax = plt.subplots(1)
        freezetime_bylap = []
        freezepercent_bylap = []
        freezeepoch_bylap = []
        for i in np.unique(lap):
            if i>0:
                assert lap.shape==beh.shape
                #remove end bit affected by filter 
                try:
                    this_beh = beh[lap==i]
                    resam = scipy.signal.decimate(this_beh, 10)
                    resam = scipy.signal.savgol_filter(resam, 50, 2)
                    end_ind = np.where(resam>0.6)[0][-1]
                    resam = resam[:end_ind]
                except Exception as e:
                    print(e)
                    # fs, ax1 = plt.subplots(1)
                    # ax1.plot(beh)
                    continue
                
                vel = np.diff(resam)
                freezeframe = np.where(vel<0.0001)[0]

                epoch_data = np.zeros_like(resam)
                epoch_data[freezeframe] = 1


                #Calculate epochs
                e=0
                epoch_length = []
                epoch_start = []
                while e<np.size(epoch_data):
                    if epoch_data[e] == 1:
                        temp_length=0
                        start_pos = e
                        while epoch_data[e]==1:
                            e+=1
                            temp_length+=1
                        if temp_length>1:
                            epoch_length.append(temp_length)
                            epoch_start.append(start_pos)
                    else:
                        e+=1

                #fix nearby epoch
                new_epoch_length = []
                diff_epoch_start = np.diff(epoch_start)
                
                e=0
                while e<np.size(diff_epoch_start)-1:
                    if diff_epoch_start[e]<=150:
                        end_pos = 0
                        start_pos = e
                        while diff_epoch_start[e]<=150 and e<np.size(diff_epoch_start)-1:
                            end_pos+=1
                            e+=1                           
                        actual_length = sum(epoch_length[start_pos:start_pos+end_pos+1])
                        new_epoch_length.append(actual_length)
                        e+=1
                    else:
                        new_epoch_length.append(epoch_length[e])
                        e+=1

                ax.plot(resam)
                ax.plot(freezeframe, resam[freezeframe], 'k.')
                ax.plot(epoch_start, np.ones_like(epoch_start), 'g*')
                print(new_epoch_length)
                print(epoch_start)
                print(epoch_length)
                ax.set_title(len(new_epoch_length))

                freezetime_bylap.append(len(freezeframe)/100)
                freezepercent_bylap.append(len(freezeframe)/len(resam)*100)
                freezeepoch_bylap.append(len(new_epoch_length))
                
        return freezetime_bylap, freezepercent_bylap, freezeepoch_bylap


