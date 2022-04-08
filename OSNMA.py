import pandas as pd
import numpy as np
from hashlib import sha256
import ecdsa
import hmac
import base64



class OSNMA():
    def __init__(self,msg_collection,num=3000):
      # OSNMA PARAMS
      self.OSNMA_LENGTH = 40
      self.KEY_SIZE = 128
      self.DS_SIZE = 512
      self.public_key = 'MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAErZl4QOS6BOJl6zeHCTnwGpmgYHEbgezdrKuYu/ghBqHcKerOpF1eEDAU1azJ0vGwe4cYiwzYm2IiC30L1EjlVQ=='
      self.kroot_key = 0
      self.sub_frame_sync = 0 # Implement to check sub-frame timing only for kroot then use it later, if sub time is same for all sats
      # Store data
      self.num=num
      self.msg_collection = msg_collection
      # State variables
      self.status = ""
      self.data_good = False
      self.kroot_good = False
      self.auth_attempts = 0
      self.chain_good = 0
      self.tag_good = 0
      self.osnma_sat_count = 0
      self.last_auth_time = 0
      self.auth_times = []
      # Process messages stored
      self.process_Pages(self.msg_collection,self.num)
      if not self.data_good:
        self.status = "Not enough data (Min 90 per satelltie).\nCurrently: {} Pages".format(len(self.sat_pages_df))
        return
      # Find root key in processed pages
      self.kroot(self.sats_OSNMA_df)
      if not self.kroot_good:
        self.status = "Currently: {} Pages\n{}\n{}\n{}".format(len(self.sat_pages_df),self.split_info, self.filter_info_str, self.kroot_info)
        return
      self.status = "Root key authenticated!\nReady to process sub-frames."

    # Takes raw SFRBX messages, converts to galileo ICD, prunes incomplete data, find OSNMA sync
    # Returns list of DataFrames with satellites currently transmitting OSNMA
    def process_Pages(self,msg_collection,num):    
      # Set initial state
      self.data_good = False
      self.sats_OSNMA_df = None
      # Process messages
      self.sat_pages_df = ublox_to_galileo(msg_collection,num)
      self.PRN_pages,self.split_info = split_df(self.sat_pages_df)
      if len(self.PRN_pages)==0:
        return
      [self.sats_OSNMA_df,self.filter_info_str] = filter_OSNMA(self.PRN_pages)
      if len(self.sats_OSNMA_df)!=0:
        self.osnma_sat_count = len(self.sats_OSNMA_df)
        self.data_good = True
    
    # Takes list of DataFrames with OSNMA sats and assembles DSM-msg, finds root key
    # Performs ECDSA Authentication of kroot key with ESA public key
    # Returns kroot content, kroot flag
    def kroot(self,sats_OSNMA_df):
      # Set initial state
      self.kroot_good=False
      # Get root key
      [kroot_content,self.kroot_info] = find_kroot(self.sats_OSNMA_df)
      # Unpack content
      [self.kroot_week,self.TOW_root,self.α,self.kroot_key,self.DSM_KROOT_msg,self.ds] = kroot_content
      if self.kroot_key == 0:
        return
      # Get ECDSA  msg, signature, and key in hex 
      [self.public_key_hex,self.msg_hex,
       self.signature_hex, self.auth_message_info] = get_auth_message(self.public_key,self.DSM_KROOT_msg,self.ds)
      # Try to authenticate root key
      try:
        self.kroot_good=authenticate_DS(self.msg_hex,self.public_key_hex,self.signature_hex)
        self.signature_info = "\nSignature Verified: " + str(self.kroot_good)
      except:
        self.signature_info = "\nSignature verification failed"

    # Get list of OSNMA satellites found and how many sub-frames exist
    def get_OSNMA_prn(self,index):
      prn = self.sats_OSNMA_df[index]['PRN'][0]
      subframes = len(self.sats_OSNMA_df[index]['PRN'])//30
      return prn,subframes
    # Get the time range of the messages stored and last authentication
    def get_duration(self):
      # Change to internal time handler
      # Current time
      curr_time=self.sat_pages_df['TOW_ms'].max()
      # Time OSNMA has been running
      run_time = curr_time - self.sat_pages_df['TOW_ms'].min()
      # TOW of latest authenticated sub-frame
      auth_age = curr_time - self.last_auth_time
      time_stats = [run_time, self.last_auth_time, curr_time, auth_age]
      time_info = "\nCurrent runtime: {} seconds\nLast Authenticated TOW: {}\nCurrent Time of Week: {}\nAuthentication Age: {} seconds".format(*time_stats)
      time_stats.append(time_info)
      return time_stats

    # Get info about OSNMA status and performance
    def get_results(self,info=True):
      result_str = "\nResult:\n KROOT Authentication (Correct/Total): {1}/{0}\n Tag Authentication (Correct/Total): {2}/{0}\n".format(self.auth_attempts,self.chain_good,self.tag_good)
      if info:
      # Combine info strings
        result_str = self.split_info + self.filter_info_str + self.kroot_info + self.signature_info + result_str + self.get_duration()[-1]
      return result_str
    # Specify id and sub-frame to authenticate chain key
    def auth_chain(self,index,sub_frame):
      # Add attempt to counter
      self.auth_attempts +=1
      chain_valid = False
      try:
        # Get sub time
        sub_time=self.sats_OSNMA_df[index]['TOW_ms'][30*sub_frame]
        # Get subweek
        gst_SF = get_sub_gst(self.sats_OSNMA_df[index],sub_time)
        gst_SF='{:032b}'.format(gst_SF)
        sub_wkn = int(gst_SF[0:12],2)
        root_age = 604800*(sub_wkn - self.kroot_week) + sub_time -self.TOW_root
        i_chain=root_age // 30 + 1
        # Get latest chain key
        [chain_key, gst_sf] = get_chain_key(self.sats_OSNMA_df[index],sub_time)
        # Run through the TESLA chain
        chain_root = prev_key_hash(chain_key,gst_sf,self.α,i_chain)
        chain_valid = chain_root == int(self.kroot_key,base=16)
        if chain_valid: 
          self.chain_good += 1
          if sub_time> self.last_auth_time:
            # Chain Key is located in the last 16 bits, last page contains 32 MACK bits
            self.auth_times.append(sub_time + 29)
            self.last_auth_time = sub_time + 29
      except: return chain_valid
      return chain_valid
    # Specify id and sub-frame to authenticate chain tag
    def auth_tag(self,index,sub_frame):
      tag_valid = False
      sub_time=self.sats_OSNMA_df[index]['TOW_ms'][30*sub_frame]
      try:
        # Calculate nav data tag
        sig=auth_tag0(self.sats_OSNMA_df[index],sub_time)
        # Get MACK tag
        [tag,tag_i_info]=get_mack_tag(self.sats_OSNMA_df[index],sub_time,0)
        tag_valid = tag==sig
        if tag_valid: self.tag_good += 1
      except: return tag_valid
      return tag_valid

def run_osnma(msg_collection, num=3000,instance= None):
  if instance == None:
    osnma_instance = OSNMA(msg_collection,num=num)
  else: osnma_instance.process_Pages(msg_collection,num)

  # Authenticate sub-frame
  if osnma_instance.data_good and osnma_instance.kroot_good:
    for index in range(osnma_instance.osnma_sat_count):
      sat_prn, sub_count = osnma_instance.get_OSNMA_prn(index)
      for i in range(1,sub_count):
        # Authenticate sub-frame
        osnma_instance.auth_chain(index,i)
        osnma_instance.auth_tag(index,i)
    return osnma_instance, osnma_instance.get_results()
  else: return osnma_instance, osnma_instance.status
  
  

# Converts u-blox pages according to ESA Galileo ICD
def ublox_to_galileo(msg_collection,num=3000): #,num=5000
  # SFRBX messages received
  sfrbx_msg=msg_collection[14][-num:]
  # Remove extra space from header info
  coloum_header= [row.split('=')[0][1:] for row in sfrbx_msg[0][1:18]]
  temp_page_list= []
  for i in range(len(sfrbx_msg)):
    # Keep only Galileo for OSNMA
    if sfrbx_msg[i][1]==' gnssId=Galileo':
      page_content = [row.split('=')[1] for row in sfrbx_msg[i][1:]]
      # Append values to last row
      temp_page_list.append(page_content)
  # Keep only E1 signals
  page_df=pd.DataFrame(temp_page_list,columns=coloum_header)
  page_df=page_df[page_df['reserved0']=='1']
  # Setup Pandas dataframe for binary pages
  df_header=['PRN', 'TOW_ms','Nav_msg']
  sat_pages_df=pd.DataFrame(columns=df_header)
  # clear temp list to hold formated binary pages
  temp_page_list= []
  # Extract u-blox data words
  data_pages=page_df[['dwrd_01','dwrd_02','dwrd_03','dwrd_04','dwrd_05','dwrd_06']].values
  meta_info=page_df[['svId','galTow']].values
  # Convert page data to binary
  for i in range(len(data_pages)):
    # Get satellite PRN of page index
    PRN = meta_info[i][0]
    # Get TOW of page index
    tow = meta_info[i][1]
    # Extract 32 bit words and convert to binary
    dwrds = [int(data_pages[i][j]) for j in range(0,6)]
    page_string = '{:032b},'*6
    data_page_bits = page_string.format(*dwrds)
    # Format as list and cut extra element
    data_page_bits=data_page_bits.split(',')[0:6]
    # dword4, cut last 8 pad.
    data_page_bits[3] = data_page_bits[3][0:-8]
    # Assembly pages according to ESA ICD
    nav_msg_even=''.join(data_page_bits[0:4])
    # Replace search and rescue, checksum bits with zeros.
    nav_msg_odd=''.join((*data_page_bits[4:6],'0'*56))
    df_row_odd = [PRN, int(tow)+1,nav_msg_odd ]
    df_row_even = [PRN, int(tow),nav_msg_even]
    # Append df rows to temp list
    temp_page_list.append(df_row_even)
    temp_page_list.append(df_row_odd)
  sat_pages_df=pd.DataFrame(temp_page_list,columns=df_header)
  return sat_pages_df

# Looks for words 2, 4, 6 to find start of sub-frame
def find_sub_start(df,drop=True):
  df = df.reset_index(drop=True)
  # Iterate until offset is found
  offset = None
  for offset in range(0,len(df)-29,2):
        # Collect Bits 2 - 8 (word type bits) from expected page 0, 2, 4 
        word_types=[int(df['Nav_msg'][page][2:8],2) for page in np.array([0,2,4,24])+offset]
        # Check if word types match expected first 3, type 2, 4, 6 and later word 5
        sync_good = word_types==[2, 4, 6, 5]
        # stop loop if words match
        if sync_good:
          break
  if drop:
    return df[offset:].reset_index(drop=True)
  return offset

# Splits dataframe into seperate dataframes for each satellite
# Removes partial sub-frames where less than 30 pages has been received
def split_df (df):
  inital_length = len(df)
  final_length = 0
  split_info = "Could not split PRN pages!"
  # Group all symbols with same TOW
  df_split=df.groupby("PRN")
  df_groups=df_split["PRN"].count()
  # PRN Received
  PRN_list=list(df_groups.index)
  # Split by Satellite PRN
  PRN_pages = []
  for PRN_sat in PRN_list:
    pages_df=df_split.get_group(PRN_sat).reset_index(drop=True)
    # Only keep satellites with at least one full sub-frame
    if len(pages_df)<30:
      continue
    good_subs = []
    # Iterate through every page looking for a complete set of 30 
    for offset in range(0,len(pages_df)-30,2):
      # Once good sub is found, skip ahead 30 pages (performance)
      if len(good_subs) != 0:
        if offset < good_subs[-1]+30:
          continue
      good_sub = find_sub_start(pages_df[offset:],drop=False)
      # Check if good sub is found
      if good_sub != None:
        good_subs.append(good_sub+offset)
    # Temp list to hold all sub-frames before merged
    subs_df = []
    for sub in good_subs:
      # Extarct expected sub-frame
      sub_section = pages_df[sub:sub+30]
      # Double check all pages are sequential, last page TOW - first page
      duration = sub_section['TOW_ms'].values[-1] - sub_section['TOW_ms'].values[0]
      if duration == 29:
        subs_df.append(sub_section)
    # Don't append if no good subs are found for satellite
    if len(subs_df)==0:
      continue
    # Merge all found sub-frames
    pages_df = pd.concat(subs_df,ignore_index=True)
    final_length += len(pages_df)
    PRN_pages.append(pages_df)
    split_info = "\nFiltering (split): Kept {}/{} pages, {:.2f}%".format(final_length,inital_length,final_length/inital_length*100)
  return PRN_pages,split_info

# Takes list of Dataframes with Galileo Sats and returns only
# Satellites actively transmitting OSNMA, and finds first and last OSNMA frame
# Minimum 60 pages per satellite
def filter_OSNMA(PRN_pages):
  sync_good = False
  sats_OSNMA_df=[]
  for sat_data in PRN_pages:
    sat_data=sat_data.reset_index(drop=True)
    if len(sat_data)>=60:
      # Check for same osnma header in batches of 30 pages
      hkroot_hdr = [sat_data['Nav_msg'][i][18:18+8] for i in range(1,len(sat_data),30)]
      # Check if any OSNMA frames are found
      if '01010010' in hkroot_hdr:
        # Keep reference list
        hkroot_hdr_good = list(hkroot_hdr)
        # Find first OSNMA Block
        first_sub = next(i for i in range(len(hkroot_hdr)) if hkroot_hdr[i] == '01010010')
        # Reverse list and find last, might cut off extra of subframe if partial end, fix in split
        hkroot_hdr.reverse()
        last_sub = next(i for i in range(len(hkroot_hdr)) if hkroot_hdr[i] == '01010010')
        last_sub = len(hkroot_hdr)-last_sub
        # Extract OSNMA section and append to OSNMA sat list
        hkroot_hdr_good = hkroot_hdr_good[first_sub : last_sub]
        sat_data = sat_data[first_sub*30 : last_sub*30].reset_index(drop=True)
        sats_OSNMA_df.append(sat_data)
  if len(sats_OSNMA_df)==0:
    return PRN_pages, "\nNo OSNMA found.\nTotal Satellite Kept: 0/{}".format(len(PRN_pages))
  # Count filtering result
  osnma_pages = 0
  split_pages = 0
  for i in range(len(sats_OSNMA_df)):
    sat_pages = len(sats_OSNMA_df[i])
    osnma_pages += sat_pages
  for i in range(len(PRN_pages)):
    sat_pages = len(PRN_pages[i])
    split_pages += sat_pages
  filter_stats = "\nFiltering (osnma): Kept {}/{} pages, {:.2f}%".format(osnma_pages,split_pages,osnma_pages/split_pages*100)
  filter_info_str = "\nFound OSNMA Satellite!\n{}\nTotal Satellite Kept: {}/{}".format(hkroot_hdr_good, len(sats_OSNMA_df),len(PRN_pages))
  return sats_OSNMA_df,filter_stats+filter_info_str

# Takes list of df with OSNMA sats and returns kroot
def find_kroot(sats_OSNMA_df,KEY_SIZE=128):
  # Group sub-frames for satellite with OSNMA
  valid_key = False
  all_blocks = False
  # Create list of list for all block ids
  dsm_blocks = [ [] for _ in range(15)]
  unique_blocks = []
  info_str = ""
  for sat in range(len(sats_OSNMA_df)):
    sat_df=sats_OSNMA_df[sat]
    df_header=['PRN', 'TOW_ms','Nav_msg']
    sub_frames =pd.DataFrame(columns=df_header)
    sat_df = sat_df.reset_index(drop=True)
    for i in range(len(sat_df)//30):
        sub_frame=sat_df[30*i:30*(i+1)]
        page_list=list(sub_frame['Nav_msg'])
        # Extract OSNMA HKROOT from odd pages
        hkroot_osnma = [page_list[i][18:18+8] for i in range(1,len(page_list),2)]
        hkroot_msg=''.join(hkroot_osnma)
        sub_frame=sub_frame[0:1]
        sub_frame['Nav_msg'] = [hkroot_msg]
        sub_frames.loc[len(sub_frames)] = sub_frame.values[0]
        # Make sure hkroot msg is good and of same type
        if hkroot_msg[0:8]=='01010010':
          block_type = int(hkroot_msg[8:12],2)
          dsm_blocks[block_type].append(hkroot_msg)
    for id in range(15):
      if dsm_blocks[id] != []:
        dsm_id = [int(dsm_blocks[id][i][12:16],base=2) for i in range(len(dsm_blocks[id]))]
        unique_blocks = np.unique(np.array(dsm_id))
        # Check if blocks 0 to 7 can be found
        all_blocks=np.array_equal(unique_blocks, np.arange(0,8))
        if all_blocks:
          # Locate blocks
          dsm_id_np=np.array(dsm_id)
          block_loc = [np.where(dsm_id_np == i)[0][0] for i in range(8)]
          # Pick out correct blocks
          DSM_KROOT = list(np.array(dsm_blocks[id])[block_loc])
          valid_key = True
          info_str = "\n\nFound root key in satellite: {}!\nLocations: {}".format(sub_frames['PRN'][0],block_loc)
          break
    if all_blocks:
      break
  # If no valid blocks are found in any satellite, cancel
  if not valid_key:
    info_str = "{}\n No valid key found.".format(unique_blocks)
    return [[0]*6,info_str]
  # Assemble KROOT msg
  DSM_KROOT_msg=[DSM_KROOT[i][16:] for i in range(len(DSM_KROOT))]  
  DSM_KROOT_msg=''.join(DSM_KROOT_msg)
  # root week number, should be around 1170-1220 for 2022
  kroot_week=int(DSM_KROOT_msg[36:48],base=2)
  # root week hour
  kroot_hour=int(DSM_KROOT_msg[48:56],base=2)
  TOW_root=kroot_hour*3600 # Time of week
  # alpha, random pattern should be '41590689997730'
  α=DSM_KROOT_msg[56:56+48]
  α=int(α,base=2)
  kroot=DSM_KROOT_msg[104:104+KEY_SIZE]
  kroot_key='{:032X}'.format(int(kroot,base=2))
  ds_start = 104+KEY_SIZE
  ds=DSM_KROOT_msg[ds_start:ds_start+512]
  # dump info
  result_info="\nRandom Pattern: {}\nKroot Key: {}\nRoot key age: Week: {}, TOW: {}\nOSNMA Digital Signature: {:0128X}".format(α,kroot_key,kroot_week,TOW_root,int(ds,2))
  info_str = info_str + result_info
  return [[kroot_week,TOW_root,α,kroot_key,DSM_KROOT_msg,ds],info_str]

# Converts public key to hex, assembles message from kroot_content
def get_auth_message(public_key,DSM_KROOT_msg,ds,KEY_SIZE=128):
  # Decode to bytes
  pub_key=base64.b64decode(public_key)
  # Encode to hex
  pub_key_hex=base64.b16encode(pub_key)
  # Reformat as hex string, and take last 130 digits (520 bits)
  public_key_hex='{:0X}'.format(int(pub_key_hex,base=16))[-130:]
  # Assemble message from DSM-KROOT
  # Always assume same header (TEMP)
  #msg_1=hkroot_hdr[0]
  msg_1='01010010'
  msg_2=DSM_KROOT_msg[8:104+KEY_SIZE]
  # Merge strings
  msg=msg_1+msg_2
  # Convert message to hex
  msg_hex='{:0X}'.format(int(msg,base=2))
  # Convert signature to hex
  signature_hex='{:0128X}'.format(int(ds,base=2))
  auth_message_info = "\nPublic Key in hex: {} \nMessage in hex: {} \nSignature in hex: {}".format(public_key_hex,msg_hex,signature_hex)
  return public_key_hex,msg_hex,signature_hex, auth_message_info

# Performs ECDSA signature authentication
# All inputs given in hex
def authenticate_DS(message,public_key,sig):
  # Load public key as verification object 
  vk = ecdsa.VerifyingKey.from_string(
      bytes.fromhex(public_key),
       curve=ecdsa.NIST256p,
        hashfunc=sha256)
  # Verify message
  result= vk.verify(
      bytes.fromhex(sig),
       bytes.fromhex(message)) # True 
  return result

# Returns GST time in 12+20 bit format from TOW
def get_sub_gst(df,sub_time):
  # Get word 5 from specified sub-frame
  word_5 = get_words(df,sub_time, words = [5])[0]
  # Read week number, TOW from word
  sub_wkn=int(word_5[-43-12:-43],2)
  tow=int(word_5[-43:-23],2)
  # Round to nearest 30 for sub-timing
  tow= round(tow/30)*30
  # GST bit format
  gst_SF='{:012b}{:020b}'.format(sub_wkn,tow)
  gst_SF=int(gst_SF,base=2)
  return gst_SF

# Calculates previous sub-frame in GST time
def prev_GST_sf (gst_time):
  if '{:032b}'.format(gst_time)[12:32] == ''.join(['0']*20):
    # Remove 1 week, add 1 week - 30 seconds (in seconds)
    gst_time=gst_time-2**20 +604770
  else: gst_time=gst_time-30
  #gst_bin= '{:032b}'.format(gst_time)
  #print([int(gst_bin[0:12],base=2),int(gst_bin[12:32],base=2)])
  return gst_time

# Retreives selected words from specified sub-frame
def get_words(df,sub_time,words=[5]):
  # Dictionary of were I/NAV words are located in nominal sub-frame
  # From Galileo Signal ICD p.34-35
  word_loc_dict = {0: 17, 1: 21, 2: 1, 3: 23, 4: 3,
                         5: 25, 6: 5, 7: 7, 8: 9, 9: 7,
                         9: 9, 16: 15}
  word_loc_dict = pd.Series(word_loc_dict)
  # Get pages from previous sub-frame, offset 1 second to get page_0
  # Locate start of sub-frame
  auth_sub_start=df[df['TOW_ms']==sub_time-30]
  # Locate end of sub-frame
  auth_sub_end=df[df['TOW_ms']==sub_time-1]
  # Collect pages
  sub_frame=df[auth_sub_start.index[0]:auth_sub_end.index[0]]
  sub_pages=list(sub_frame['Nav_msg'])
  # Add blank page to get page count to match
  sub_pages.insert(0, '0'*120)
  # Extract bits at word (1/2) and word (2/2) from all pages
  word_1_2 = [sub_pages[i][2:114] for i in range(len(sub_pages))]
  word_2_2 = [sub_pages[i][2:18] for i in range(len(sub_pages))]
  # Extract selected words¨
  page_locations = list(word_loc_dict[words])
  words = [word_1_2[i] + word_2_2[i+1] for i in page_locations]
  return words

# Returns the complete MACK message, starting at sub_time
def get_mack_msg(df,sub_time):
  df=df.reset_index(drop=True)
  # Get MACK Message from next sub-frame
  auth_sub_start=df[df['TOW_ms']==sub_time]
  # Get time of last page in sub-frame
  auth_sub_end=df[df['TOW_ms']==sub_time+29]
  # Collect pages, add 1 second to include last page even if next sub does not exist
  sub_frame=df[auth_sub_start.index[0]:auth_sub_end.index[0]+1]
  sub_pages=list(sub_frame['Nav_msg'])
  # Extract mack bits from odd pages
  mack_msg = [sub_pages[i][18+8:26+32] for i in range(1,len(sub_pages),2)]
  mack_sub=''.join(mack_msg)
  return mack_sub

# Returns the MACK TESLA chain key as integer for sub-frame which starts at sub_time.
def get_chain_key (df,sub_time,KEY_SIZE=128): 
  mack_sub=get_mack_msg(df,sub_time)
  # Extract key from mack message
  chain_key_bits=mack_sub[336:336+KEY_SIZE]
  # Convert binary to integer
  chain_key_int=int(chain_key_bits,base=2)
  # Convert to hex for display
  chain_key='{:032X}'.format(int(chain_key_bits,base=2))
  gst_SF = get_sub_gst(df,sub_time)
  return chain_key_int, gst_SF

# chain key as integer or hex, gst time as 32 bits (12 bits week number, 20 bits time of week)
def prev_key_hash(chain_key,gst_time_sf,α, index=1):
  # Enforce max length, 2 hours old chain key, normally replaced every hour
  if index<0 or index>250:
    print("ROOT KEY TOO OLD! {} sub-frames Not valid, max 250, aborting!".format(index))
    return None
  if type(chain_key) == str:
    chain_key = int(chain_key,base=16)
  for i in range(index):
    # Roll back 30 seconds
    gst_time_sf=prev_GST_sf(gst_time_sf)
    # Assemble hash input in bits
    hash_bits='{:0128b}{:032b}{:048b}'.format(chain_key,gst_time_sf,α)
    # Convert to hexadecimal 
    hex_input='{:052X}'.format(int(hash_bits,base=2))
    # Do sha-256
    out=sha256(bytes.fromhex(hex_input)).hexdigest()[0:32]
    # Take first 32 digits in hex (128 bits)
    chain_key = int(out[0:32],base=16)
  return chain_key

# Retrivies chain key and nav data from OSNMA dataframe
# Calculates nav data tag
def auth_tag0(df,sub_time,ADKD=0):
  # Key from subframe after nav data
  # Galileo system time of the sub-frame with the tag
  [chain_key,gst_sf] = get_chain_key(df,sub_time+30)
  # Get the start of subframe instead of end.
  gst_sf = gst_sf - 30
  # ADKD = 0, Ephemeris, Clock and Status, I/NAV Word 1 - 5 (E1b, E5b)
  words = get_words(df,sub_time,words = list(range(1,6)))
  # Trim words according to OSNMA ICD
  words[0] = words[0][6:-2]   # 120 bits from word 1, cut first 6, last 2
  words[1] = words[1][6:-2]   # 120 bits from word 2, cut first 6, last 2
  words[2] = words[2][6: ]    # 122 bits from word 3, cut first 6
  words[3] = words[3][6:-2]   # 120 bits from word 4, cut first 6, last 2
  words[4] = words[4][6:-55]  # 67 bits from word 5, cut first 6, last 55
  # Total 549 bits, join words and convert to integer
  adkd0 = int(''.join(words),base=2)
  # 8 bit PRN of authenticating satellite 
  prn_d = int(df["PRN"][0])
  #8 bit tag position, CTR always equals 1 for Tag0.
  ctr=1
  # nmas, status of OSNMA, '01' = TEST PHASE
  nmas= int('01',base=2)
  # 1 padding bit required for ADKD0
  padding = 0
  # Assemble hash message
  hash_bits='{:08b}{:032b}{:08b}{:02b}{:0549b}{:01b}'.format(prn_d,gst_sf,ctr,nmas,adkd0,padding)
  # Convert to hex
  msg='{:0150X}'.format(int(hash_bits,base=2))
  chain_key='{:032X}'.format(chain_key)
  # HMAC sha256
  signature = hmac.new(
      bytes.fromhex(chain_key),
       msg = bytes.fromhex(msg),
        digestmod = sha256
        ).hexdigest().upper()
  # truncate, 40 bits, 10 hex digits
  signature = signature[0:10]
  return signature
# Retrivies Mack tag from OSNMA dataframe at specified sub-frame and tag index
def get_mack_tag (df,sub_time,tag_index):
  mack_sub=get_mack_msg(df,sub_time)
  # tag size 40 bits, 16 info bits
  TAG_SIZE = 40
  TAG_INFO = 16
  tag_i=mack_sub[(TAG_SIZE+TAG_INFO)*tag_index:(TAG_SIZE+TAG_INFO)*(tag_index+1)]
  tag_i_info=tag_i[-16:]
  tag_i=tag_i[0:40]
  tag_i='{:010X}'.format(int(tag_i,base=2))
  return [tag_i,tag_i_info]