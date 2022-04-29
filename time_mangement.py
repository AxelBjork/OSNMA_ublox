import socket
import pandas as pd
import matplotlib.pyplot as plt

# Get time from latest '<UBX(NAV-TIMEGAL' message
# Adjust offset for spoofed gnss data
def get_gnss_time(gal_time_msg,offset = 0):
  galileo_time = gal_time_msg[-1][2] if len(gal_time_msg[-1:]) else " galTow=0"
  galileo_time =float(galileo_time.split('=')[1])
  return galileo_time + offset

# Get NTP Time
# Set offset to adjust for old (real) GNSS data
def get_ntp_time(offset = 0):
  # reference time (in seconds since 1900-01-01 00:00:00)
  TIME1970 = 2208988800 # 1970-01-01 00:00:00
  # List of common NTP servers 
  hosts = ['time.windows.com','0.de.pool.ntp.org','europe.pool.ntp.org']
  # Setup server
  client = socket.socket( socket.AF_INET, socket.SOCK_DGRAM)
  # Set 2 second timeout limit
  client.settimeout(2)
  msg = '\x1b' + 47 * '\0'
  try:
    client.sendto(msg.encode('utf-8'), (hosts[0],123))
    msg, _ = client.recvfrom( 1024 ) # buffer 1024 
    # Convert bytes to integer, UTC time from NTP time
    ntp_time = int.from_bytes(msg[-8:], 'big') / 2 ** 32 - TIME1970
    # Calculate TOW with offset for simulation
    ntp_tow = (ntp_time +offset) % 604800
    return ntp_tow
  except:
      return 0

class Time_Handler():
    def __init__(self):
      # Implemented as TOW for demo
      self.current_time = 0
      self.gnss_delta = 0
      self.gnss_time = 0 
      self.sync_age = 9999
      self.npt_requests = 0
      self.npt_offset = 0 # For old file playbacksd
      self.gnss_offset = 0 # For spoofed gps simulation
      self.time_source = "None"
      # OSNMA 
      self.OSNMA_authenticated = False
      self.OSNMA_sync = False
      self.OSNMA_fast_req = 30
      self.OSNMA_slow_req = 300

    # Set time according to NTP
    def ntp_update(self):
      self.npt_requests +=1
      # Get time in seconds from NTP server
      ntp_time = get_ntp_time(offset = self.npt_offset)
      if ntp_time != 0:
        self.time_source = "NTP"
        # Set time since sync 0 seconds
        self.sync_age = 0
        # Set current time (only time of week!)
        self.current_time = ntp_time
        
      # Update internally if no internet connection
      else: self.internal_update()

    # Set time according to GNSS
    def gnss_update(self,gal_time_msg):
      # Get time from latest '<UBX(NAV-TIMEGAL' message
      galileo_time = get_gnss_time(gal_time_msg,self.gnss_offset)
      if galileo_time != 0:
        self.time_source = "GNSS"
        # Set time since sync 0 seconds
        self.sync_age = 0
        # Set current time 
        self.current_time = galileo_time
      else: self.internal_update()

    # Update time by 1 second using simulated RTC
    def internal_update(self,noise=True):
      # Consider GNSS time as valid within sync, always keep NTP time
      if not self.OSNMA_sync and self.time_source == "GNSS":
        self.time_source = "None"
      if noise:
        # Simulate a slow running clock
        step = 0.85
      else: step = 1
      # Increment time
      self.current_time +=step
      # Increment sync age
      self.sync_age += 1

    # Call once per second 
    def update_time(self,gal_time_msg):
      # Read gps and log delta
      self.run_time = len(gal_time_msg)
      self.gnss_time = get_gnss_time(gal_time_msg,self.gnss_offset)
      self.gnss_delta = abs(self.gnss_time - self.current_time)
      self.OSNMA_sync = self.gnss_delta <= self.OSNMA_fast_req
      # Check if GNSS is authenticated
      if self.OSNMA_authenticated:
        # Check if GNSS is already being used and still valid
        if self.OSNMA_sync:
          # Fix so only uses auth sub-frame time instead of gnss receiver time
          # Update time according to GNSS (Nominal)
          self.gnss_update(gal_time_msg)
        # Do NTP sync
        elif  self.sync_age > 200:
          self.ntp_update()
        # Else rely on internal RTC
        else: self.internal_update()
      # Without OSNMA, update every hour
      elif self.sync_age > 200: # 60*60*24*14 (14 days)
        self.ntp_update()
      else: self.internal_update()  # drifts 50 seconds / month
      # Update delta
      self.gnss_delta = abs(self.gnss_time - self.current_time)
      
    # Returns current time
    def get_time(self):
      info = [self.run_time,self.current_time,self.gnss_time, self.gnss_delta, 
              self.sync_age,self.OSNMA_authenticated, self.time_source]
      return info


def plot_time(df,ntp_syncs,gnss_syncs):
  font = {'family' : 'DejaVu Sans',
          'weight' : 'normal',
          'size'   : 7}
  plt.rc('font', **font)
  plt.rcParams['figure.dpi'] = 150
  plt.rcParams["figure.figsize"] = [6.0, 2.5] 
  # Plot time data
  fig, axes = plt.subplots(1, 1)
  df[["Handler Time","GNSS Time",]].plot(ax = axes)
  df[["Reference Time"]].plot(ax = axes,alpha=0.3,color='black')

  axes.set_title('Time Handler Timeline', fontweight ="bold")
  plt.xlabel('Time Step [sec]')

  #plt.title('U-blox L1 Signal Metrics', fontweight ="bold")
  plt.ylabel('Receiver Time of Week [sec]')

  # Hide mark labels at negative x
  axes.axvspan(-11, -10, facecolor='g', alpha=0.3, label = "OSNMA Authenticated")
  axes.axvspan(-11, -10, facecolor='r', alpha=0.9, label = "NTP Request(s)")

  # Check NTP and GNSS sync times
  for index in ntp_syncs.index:
    axes.axvspan(index-1.5, index, facecolor='r', alpha=0.9)
  # Fix retrospective syncs
  for index in gnss_syncs.index:
    axes.axvspan(index-29, index, facecolor='g', alpha=0.3)

  plt.legend()
  # Mark spoofing secions
  fig.patch.set_alpha(1)
  plt.xlim(-5,len(df))
  plt.ylim(df["GNSS Time"].max() - 1.1*len(df) ,df["GNSS Time"].max())
  return fig


  
def run_time(osnma_instance,msg_collection,max_dur=500,plot=True):
  # Authenticated sub-frames 
  #root_key_time = 385922

  # Extract Galileo clock messages 
  gal_time_ref= msg_collection[11][-max_dur:]
  # Set up time handler
  time_instance = Time_Handler()
  # Calculate offset for recorded file 
  gnss_offset = get_gnss_time(gal_time_ref[:1]) - get_ntp_time()
  receiver_time = []
  for step in range(1,len(gal_time_ref)):
    # Set offset to simulate matching NTP time
    time_instance.npt_offset = gnss_offset + step-1
    time_instance.OSNMA_authenticated = False
    # Send list with one more message per loop
    gal_time_msg= gal_time_ref[0:step]
    # Check if OSNMA was authenticated this step
    gal_time = int(get_gnss_time(gal_time_msg))
    if gal_time in osnma_instance.auth_times[2:]:# Assume root key is found on 3rd sub-frame
      time_instance.OSNMA_authenticated = True
    if step==305: # fix for recording crash
      time_instance.gnss_offset = -58 # 42
    # update time with OSNMA status
    time_instance.update_time(gal_time_msg)
    receiver_time.append(time_instance.get_time())
  # Create dataframe and read 
  df = pd.DataFrame(receiver_time,columns=["Reference Time","Handler Time","GNSS Time","GNSS Time Delta","Sync age","OSNMA_Auth","Time Source"])
  # Set reference time
  df["Reference Time"] = df["Reference Time"]+df["Handler Time"][0]
  # Check when root key was found (manual)
  # Check external syncs
  external_syncs = df[df["Sync age"] == 0]
  ntp_syncs = external_syncs[external_syncs["Time Source"] == "NTP"]
  gnss_syncs = external_syncs[external_syncs["Time Source"] == "GNSS"]
  uptime = round(len(gnss_syncs) * 30 / len(df),3)*100
  time_info = "\nTime Handler ran for {} Seconds\nTotal NTP requests: {}\nValid OSNMA sub-frames: {}\nGNSS Sync uptime: {}%\n".format(len(df),len(ntp_syncs),len(gnss_syncs),uptime)
  time_fig = plot_time(df,ntp_syncs,gnss_syncs) if plot else ""
  return time_instance, df, time_info, time_fig