import pandas as pd
import numpy as np


# Split up rawx message into individual satellites
def rawx_struct(rawx_msg):
  rawx_sats=''.join(rawx_msg).split('reserved3_')
  rawx_header=rawx_sats[0].split(' ')[0:9]
  rawx_sats[0]=rawx_sats[0].split(' ')[8:-1]
  # Split data for each satellite
  for i in range(1,len(rawx_sats)):
    rawx_sats[i] =rawx_sats[i].split(' ')[1:-1]
  # Cut empty last row
  rawx_sats=rawx_sats[0:-1]
  return [rawx_header,rawx_sats]

# Process the latest RXM-RAWX and retreive latest CNO, lock time, satellites connected. 
def get_latest_signal(msg_collection):
  if len(''.join(msg_collection[12][-1]).split('reserved3_'))==1:
    return "No Satellites Connected!"
  else:
    [rawx_header,rawx_sats]=rawx_struct(msg_collection[12][-1])
    rcTOW=round(float(rawx_header[1].split('=')[1]))
    coloum_header= [row.split('=')[0] for row in rawx_sats[0]]
    temp_page_list= []
    for sat in rawx_sats:
        # Extract sat parameters
        sat_row = [row.split('=')[1] for row in sat]
        sat_row[:3] = [float(value) for value in sat_row[:3]]
        sat_row[4:] = [int(value) for value in sat_row[4:]]
        # Append values to last row
        temp_page_list.append(sat_row)
    rawx_df=pd.DataFrame(temp_page_list,columns=coloum_header)
    # Count unique satellites by constellation
    sat_count = rawx_df.groupby("gnssId_01")['gnssId_01'].count()
    sat_count = [str(sat_count.index[i])+', '+ str(sat_count[i]) for i in range(len(sat_count))]
    # Only L1 signals
    rawx_df_L1=rawx_df[rawx_df['reserved2_01']==0]
    sat_count_L1 = rawx_df_L1.groupby("gnssId_01")['gnssId_01'].count()
    sat_count_L1 = [str(sat_count_L1.index[i])+', '+ str(sat_count_L1[i]) for i in range(len(sat_count_L1))]
    # Carrier to noise ratio L1
    avg_CNO=round(rawx_df_L1['cno_01'].mean(),1)
    # Satellite Lock duration L1
    avg_lock_time=np.round(rawx_df_L1['locktime_01'].mean(skipna=True))
    sat_info_str="\nAverage CNO (L1): {}\nAverage Lock duration (L1): {}\n\nSatellites Connected:{}\n Of which L1/E1: {}".format(avg_CNO,avg_lock_time,sat_count,sat_count_L1)
    return sat_info_str

# Around 6 ms execution time for 10 min log
def read_live_data(msg_collection):
  latest = [[] for _ in range(len(msg_collection))]
  for i in range(len(msg_collection)):
    latest[i] =  msg_collection[i][-5:]
    latest[i].reverse()
  
  # Get params from MON-RF
  AGC_L1 = [latest[6][i][11] for i in range(len(latest[6]))]
  jam_ind = [latest[6][i][12] for i in range(len(latest[6]))]
  jam_state = [latest[6][i][5] for i in range(len(latest[6]))]
  # Spoof state from NAV-STATUS
  spoof_state = [latest[13][i][11] for i in range(len(latest[13]))]
  # Clock Bias
  iTOW_bias = [latest[8][i][2] for i in range(len(latest[8]))]
  # Receiver time
  rcvToW = [latest[12][i][1].split('=')[1] for i in range(len(latest[12]))]
  # GPS Time
  iTOW = [latest[8][i][1] for i in range(len(latest[8]))]
  # Position
  position = [latest[2][i][2]+latest[2][i][4] for i in range(len(latest[2]))]
  # Assembly stats
  latest_status = [AGC_L1,jam_ind,jam_state,spoof_state,iTOW_bias,rcvToW,iTOW,position]
  # Adjust some values to only keep last instead of 5 last
  latest_status[-3]= round(float(latest_status[-3][0]))
  latest_status[-2]= latest_status[-2][0]
  latest_status[-1]= latest_status[-1][0]
  # Get satellite info and add to collection
  sat_info=get_latest_signal(latest)
  latest_status.append(sat_info)
  # Reverse order
  data_dump_str=" {} \n {}\n {}\n {}\n {}\n Time: {}, {}\n Position:{}\n {}".format(*latest_status)
  return data_dump_str