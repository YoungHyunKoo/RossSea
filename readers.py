#Import necesary modules
#Use shorter names (np, pd, plt) instead of full (numpy, matplotlib.pylot) for convenience when using

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import utils as ut
import pandas as pd
import h5py
import os
import xarray as xr
import datetime as dt
from astropy.time import Time
from sklearn.neighbors import KernelDensity

def convert_time(delta_time):
    times = []
    years = []
    months = []
    days = []
    hours = []
    minutes = []
    seconds =[]
    
    for i in range(0, len(delta_time)):
        times.append(dt.datetime(1980, 1, 6) + dt.timedelta(seconds = delta_time[i]))
        years.append(times[i].year)
        months.append(times[i].month)
        days.append(times[i].day)
        hours.append(times[i].hour)
        minutes.append(times[i].minute)
        seconds.append(times[i].second)
    
    temp = pd.DataFrame({'time':times, 'year': years, 'month': months, 'day': days,
                         'hour': hours, 'minute': minutes, 'second': seconds
                        })
    return temp

# Function to calculate the number of the same photon pulse id
def count_pid(pid):
    idcount = np.zeros(len(pid))
    firstid = pid[0]
    count = 0
    for i in range(0, len(pid)):
        if pid[i] == firstid:
            count += 1
        else:
            idcount[i-count:i] = count
            firstid = pid[i]
            count = 1
    idcount[i-count:i] = count
    return idcount

# Function to read ATL03 data (.h5 format)
def getATL03(fname, beam_number):
    # 0, 2, 4 = Strong beam; 1, 3, 5 = weak beam
    
    f = h5py.File(fname, 'r')
    
    orient = f['orbit_info']['sc_orient'][:]  # orientation - 0: backward, 1: forward, 2: transition
    
    if len(orient) > 1:
        print('Transitioning, do not use for science!')
        return [[] for i in beamlist]
    elif (orient == 0):
        beams=['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']                
    elif (orient == 1):
        beams=['gt3r', 'gt3l', 'gt2r', 'gt2l', 'gt1r', 'gt1l']
    # (strong, weak, strong, weak, strong, weak)
    # (beam1, beam2, beam3, beam4, beam5, beam6)

    beam = beams[beam_number]    
    
    # height of each received photon, relative to the WGS-84 ellipsoid (with some, not all corrections applied, see background info above)
    heights=f[beam]['heights']['h_ph'][:]
    # latitude (decimal degrees) of each received photon
    lats=f[beam]['heights']['lat_ph'][:]
    # longitude (decimal degrees) of each received photon
    lons=f[beam]['heights']['lon_ph'][:]
    # seconds from ATLAS Standard Data Product Epoch. use the epoch parameter to convert to gps time
    deltatime=f[beam]['heights']['delta_time'][:]
    # confidence level associated with each photon event
    # -2: TEP
    # -1: Events not associated with a specific surface type
    #  0: noise
    #  1: buffer but algorithm classifies as background
    #  2: low
    #  3: medium
    #  4: high
    # Surface types for signal classification confidence
    # 0=Land; 1=Ocean; 2=SeaIce; 3=LandIce; 4=InlandWater    
    conf=f[beam]['heights']['signal_conf_ph'][:,2] #choose column 2 for confidence of sea ice photons
    # number of ATL03 20m segments
    n_seg, = f[beam]['geolocation']['segment_id'].shape
    # first photon in the segment (convert to 0-based indexing)
    Segment_Index_begin = f[beam]['geolocation']['ph_index_beg'][:] - 1
    # number of photon events in the segment
    Segment_PE_count = f[beam]['geolocation']['segment_ph_cnt'][:]
    # along-track distance for each ATL03 segment
    Segment_Distance = f[beam]['geolocation']['segment_dist_x'][:]
    # along-track distance (x) for photon events
    x_atc = np.array(f[beam]['heights']['dist_ph_along'][:])
    # cross-track distance (y) for photon events
    y_atc = np.array(f[beam]['heights']['dist_ph_across'][:])
    # The photon event counter is part of photon ID and counts from 1 for each channel until reset by laser pulse counter
    pulse_id = f[beam]['heights']['ph_id_pulse'][:]
    # pcount: number of the returned photon that has the same pulse id
    pulse_count = count_pid(pulse_id)
    
    ph_id_count = f[beam]['heights']['ph_id_count'][:]
    
    # Remove the uneffective reference photons (no geo-correction parameters)
    mask_ind = (Segment_Index_begin >= 0)
    Segment_Index_begin = Segment_Index_begin[mask_ind]
    Segment_PE_count = Segment_PE_count[mask_ind]
    n_seg = len(Segment_PE_count)
    
    # Geographical correction parameters (refer to the ATL03 documents)
    geocorr0 = f[beam]['geophys_corr/dac'][:] + f[beam]['geophys_corr/tide_earth'][:] + \
    f[beam]['geophys_corr/geoid'][:] +  f[beam]['geophys_corr/tide_load'][:] + f[beam]['geophys_corr/tide_ocean'][:] + \
    f[beam]['geophys_corr/tide_pole'][:] + f[beam]['geophys_corr/tide_oc_pole'][:]
    
    # Remove unaffective geo-correction values
    geocorr0 = geocorr0[geocorr0 != np.inf]
    
    # Since the number of reference points are less than the original photons,
    # reference heights of all photons should be interpolated from the existing reference points
    geocorr = np.zeros(len(x_atc))

    # Delta time to gps seconds
    atlas_epoch=f[beam]['/ancillary_data/atlas_sdp_gps_epoch'][0]
    temp = convert_time(deltatime + atlas_epoch)

    for j in range(n_seg):
        # index for 20m segment j
        idx = Segment_Index_begin[j]
        # number of photons in 20m segment
        cnt = Segment_PE_count[j]
        # add segment distance to along-track coordinates
        x_atc[idx:idx+cnt] += Segment_Distance[j]        

        for k in range(idx, idx+cnt):
            if j < n_seg-1:
                ratio = (x_atc[k] - x_atc[idx]) / (x_atc[idx+cnt] - x_atc[idx])
                geocorr[k] =geocorr0[j+1] + (geocorr0[j+1] - geocorr0[j]) * ratio
            else:
                geocorr[k] = geocorr0[j]
        
    df03=pd.DataFrame({'beam': beam, 'lat':lats,'lon':lons,'x':x_atc,'y':y_atc,
                       'heights':heights, 'geocorr':geocorr,
                       'elev': heights-geocorr, 'deltatime':deltatime,'conf':conf,
                       'ph_id_count': ph_id_count, 'pulse_count': pulse_count})
    
    # Concatenate ATL03 dataframe and time dataframe
    df03 = pd.concat([df03, temp], axis=1).reset_index(drop = True)
    
    return df03


def getATL03dict(FILENAME, ATTRIBUTES=True, VERBOSE=False):
    """ Dictionary ATL03 reader
    Created by the NASA GSFC Python 2018

	This is a more complex/sophisticated reader, using Python dictionaries to store/manage the ATL03 variables
    
	Args:
		FIELNAME (str): File path of the ATL03 dataset
		ATTRIBUTES (flag): if true store the ATL03 attributes
		VERBOSE (flag): if true output HDF5 file information

	returns:
		Python dictionary of variables and also a list of variables

	"""
    
    #-- Open the HDF5 file for reading
    fileID = h5py.File(os.path.expanduser(FILENAME), 'r')

    #-- Output HDF5 file information
    if VERBOSE:
        print(fileID.filename)
        print(list(fileID.keys()))

    #-- allocate python dictionaries for ICESat-2 ATL03 variables and attributes
    IS2_atl03_mds = {}
    IS2_atl03_attrs = {} if ATTRIBUTES else None

    #-- read each input beam within the file
    IS2_atl03_beams = [k for k in fileID.keys() if bool(re.match('gt\d[lr]',k))]
    for gtx in IS2_atl03_beams:
        IS2_atl03_mds[gtx] = {}
        IS2_atl03_mds[gtx]['heights'] = {}
        IS2_atl03_mds[gtx]['geolocation'] = {}
        IS2_atl03_mds[gtx]['bckgrd_atlas'] = {}
        IS2_atl03_mds[gtx]['geophys_corr'] = {}
        #-- get each HDF5 variable
        #-- ICESat-2 Measurement Group
        for key,val in fileID[gtx]['heights'].items():
            IS2_atl03_mds[gtx]['heights'][key] = val[:]
        #-- ICESat-2 Geolocation Group
        for key,val in fileID[gtx]['geolocation'].items():
            IS2_atl03_mds[gtx]['geolocation'][key] = val[:]
        #-- ICESat-2 Background Photon Rate Group
        for key,val in fileID[gtx]['bckgrd_atlas'].items():
            IS2_atl03_mds[gtx]['bckgrd_atlas'][key] = val[:]
        #-- ICESat-2 Geophysical Corrections Group: Values for tides (ocean,
        #-- solid earth, pole, load, and equilibrium), inverted barometer (IB)
        #-- effects, and range corrections for tropospheric delays
        for key,val in fileID[gtx]['geophys_corr'].items():
            IS2_atl03_mds[gtx]['geophys_corr'][key] = val[:]

        #-- Getting attributes of included variables
        if ATTRIBUTES:
            #-- Getting attributes of IS2_atl03_mds beam variables
            IS2_atl03_attrs[gtx] = {}
            IS2_atl03_attrs[gtx]['heights'] = {}
            IS2_atl03_attrs[gtx]['geolocation'] = {}
            IS2_atl03_attrs[gtx]['bckgrd_atlas'] = {}
            IS2_atl03_attrs[gtx]['geophys_corr'] = {}
            IS2_atl03_attrs[gtx]['Atlas_impulse_response'] = {}
            #-- Global Group Attributes
            for att_name,att_val in fileID[gtx].attrs.items():
                IS2_atl03_attrs[gtx][att_name] = att_val
            #-- ICESat-2 Measurement Group
            for key,val in fileID[gtx]['heights'].items():
                IS2_atl03_attrs[gtx]['heights'][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl03_attrs[gtx]['heights'][key][att_name]=att_val
            #-- ICESat-2 Geolocation Group
            for key,val in fileID[gtx]['geolocation'].items():
                IS2_atl03_attrs[gtx]['geolocation'][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl03_attrs[gtx]['geolocation'][key][att_name]=att_val
            #-- ICESat-2 Background Photon Rate Group
            for key,val in fileID[gtx]['bckgrd_atlas'].items():
                IS2_atl03_attrs[gtx]['bckgrd_atlas'][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl03_attrs[gtx]['bckgrd_atlas'][key][att_name]=att_val
            #-- ICESat-2 Geophysical Corrections Group
            for key,val in fileID[gtx]['geophys_corr'].items():
                IS2_atl03_attrs[gtx]['geophys_corr'][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl03_attrs[gtx]['geophys_corr'][key][att_name]=att_val

    #-- ICESat-2 spacecraft orientation at time
    IS2_atl03_mds['orbit_info'] = {}
    IS2_atl03_attrs['orbit_info'] = {} if ATTRIBUTES else None
    for key,val in fileID['orbit_info'].items():
        IS2_atl03_mds['orbit_info'][key] = val[:]
        #-- Getting attributes of group and included variables
        if ATTRIBUTES:
            #-- Global Group Attributes
            for att_name,att_val in fileID['orbit_info'].attrs.items():
                IS2_atl03_attrs['orbit_info'][att_name] = att_val
            #-- Variable Attributes
            IS2_atl03_attrs['orbit_info'][key] = {}
            for att_name,att_val in val.attrs.items():
                IS2_atl03_attrs['orbit_info'][key][att_name] = att_val

    #-- number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    #-- and ATLAS Standard Data Product (SDP) epoch (2018-01-01:T00:00:00Z UTC)
    #-- Add this value to delta time parameters to compute full gps_seconds
    #-- could alternatively use the Julian day of the ATLAS SDP epoch: 2458119.5
    #-- and add leap seconds since 2018-01-01:T00:00:00Z UTC (ATLAS SDP epoch)
    IS2_atl03_mds['ancillary_data'] = {}
    IS2_atl03_attrs['ancillary_data'] = {} if ATTRIBUTES else None
    for key in ['atlas_sdp_gps_epoch']:
        #-- get each HDF5 variable
        IS2_atl03_mds['ancillary_data'][key] = fileID['ancillary_data'][key][:]
        #-- Getting attributes of group and included variables
        if ATTRIBUTES:
            #-- Variable Attributes
            IS2_atl03_attrs['ancillary_data'][key] = {}
            for att_name,att_val in fileID['ancillary_data'][key].attrs.items():
                IS2_atl03_attrs['ancillary_data'][key][att_name] = att_val

    #-- channel dead time and first photon bias derived from ATLAS calibration
    cal1,cal2 = ('ancillary_data','calibrations')
    for var in ['dead_time','first_photon_bias']:
        IS2_atl03_mds[cal1][var] = {}
        IS2_atl03_attrs[cal1][var] = {} if ATTRIBUTES else None
        for key,val in fileID[cal1][cal2][var].items():
            #-- get each HDF5 variable
            if isinstance(val, h5py.Dataset):
                IS2_atl03_mds[cal1][var][key] = val[:]
            elif isinstance(val, h5py.Group):
                IS2_atl03_mds[cal1][var][key] = {}
                for k,v in val.items():
                    IS2_atl03_mds[cal1][var][key][k] = v[:]
            #-- Getting attributes of group and included variables
            if ATTRIBUTES:
                #-- Variable Attributes
                IS2_atl03_attrs[cal1][var][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl03_attrs[cal1][var][key][att_name] = att_val
                if isinstance(val, h5py.Group):
                    for k,v in val.items():
                        IS2_atl03_attrs[cal1][var][key][k] = {}
                        for att_name,att_val in val.attrs.items():
                            IS2_atl03_attrs[cal1][var][key][k][att_name]=att_val

    #-- get ATLAS impulse response variables for the transmitter echo path (TEP)
    tep1,tep2 = ('atlas_impulse_response','tep_histogram')
    IS2_atl03_mds[tep1] = {}
    IS2_atl03_attrs[tep1] = {} if ATTRIBUTES else None
    for pce in ['pce1_spot1','pce2_spot3']:
        IS2_atl03_mds[tep1][pce] = {tep2:{}}
        IS2_atl03_attrs[tep1][pce] = {tep2:{}} if ATTRIBUTES else None
        #-- for each TEP variable
        for key,val in fileID[tep1][pce][tep2].items():
            IS2_atl03_mds[tep1][pce][tep2][key] = val[:]
            #-- Getting attributes of included variables
            if ATTRIBUTES:
                #-- Global Group Attributes
                for att_name,att_val in fileID[tep1][pce][tep2].attrs.items():
                    IS2_atl03_attrs[tep1][pce][tep2][att_name] = att_val
                #-- Variable Attributes
                IS2_atl03_attrs[tep1][pce][tep2][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl03_attrs[tep1][pce][tep2][key][att_name] = att_val

    #-- Global File Attributes
    if ATTRIBUTES:
        for att_name,att_val in fileID.attrs.items():
            IS2_atl03_attrs[att_name] = att_val

    #-- Closing the HDF5 file
    fileID.close()
    #-- Return the datasets and variables
    return (IS2_atl03_mds,IS2_atl03_attrs,IS2_atl03_beams)



def getATL03xr(fileT, beamStr, groupStr='sea_ice_segments', fileinfo=False):
    """ xarray ATL03 reader
    NOT WORKING YET
    Written by Alek Petty, June 2018 (alek.a.petty@nasa.gov)

	This approach is very easy! Maybe not everyone is used to xarray is theonly real downside..
    
	Args:
		fileT (str): File path of the ATL03 dataset
		beamStr (str): ICESat-2 beam (the number is the pair, r=strong, l=weak)
        groupStr (str): subgroup of data in the ATL07 file we want to extract.

	returns:
        xarrray dataframe

	"""
    dsHeights = xr.open_dataset(fileT,group='/'+beamStr+'/heights/')
    #dalons = xr.DataArray(dsHeights.coords['lon_ph'].values,
    #              dims=['date_time'],
    #              coords={dsHeights.coords['lon_ph']})
    # Add delta time as a variable
    
    dsHeightspd = dsHeights.to_dataframe()
    dsHeightspd.reset_index()
    #dsHeightspd['delta_time']=dsHeights.coords['delta_time'].values
    #dsHeights['lats']=dsHeights.coords['lat_ph'].values
    # The height segment ID is a much better index/dimension (as delta_time has some duplicates)
    # Need to do this before merging the datasets
    #dsMain = dsMain.swap_dims({'delta_time': 'height_segment_id'})
    
    # Merge the datasets
    #dsAll=xr.merge([dsHeights, dsMain])
    
    #dsHeights = dsHeights.swap_dims({'delta_time': 'height_segment_id'})
    #calDates=dsMain.coords['delta_time'].values
    
    if fileinfo==True:
        f = h5py.File(dataFile,'r')
        # print group info
        groups = list(f.keys())
        for g in groups:
            group = f[g]
            if printGroups:
                print('---')
                print('Group: {}'.format(g))
                print('---')
                for d in group.keys():
                    print(group[d])
    return dsHeightspd

def getATL07xr(fileT, beamStr, groupStr='sea_ice_segments', fileinfo=False):
    """ xarray ATL07 reader
    Written by Alek Petty, June 2018 (alek.a.petty@nasa.gov)

	This approach is very easy! Maybe not everyone is used to xarray is theonly real downside..
    
	Args:
		fileT (str): File path of the ATL07 dataset
		beamStr (str): ICESat-2 beam (the number is the pair, r=strong, l=weak)
        groupStr (str): subgroup of data in the ATL07 file we want to extract.

	returns:
        xarrray dataframe

	"""
    dsMain = xr.open_dataset(fileT,group='/'+beamStr+'/'+'sea_ice_segments')
    dsHeights = xr.open_dataset(fileT,group='/'+beamStr+'/'+'sea_ice_segments/heights/')
    
    # The height segment ID is a much better index/dimension (as delta_time has some duplicates)
    # Need to do this before merging the datasets
    dsMain = dsMain.swap_dims({'delta_time': 'height_segment_id'})
    
    # Merge the datasets
    dsAll=xr.merge([dsHeights, dsMain])
    
    #dsHeights = dsHeights.swap_dims({'delta_time': 'height_segment_id'})
    #calDates=dsMain.coords['delta_time'].values
    
    if fileinfo==True:
        f = h5py.File(dataFile,'r')
        # print group info
        groups = list(f.keys())
        for g in groups:
            group = f[g]
            if printGroups:
                print('---')
                print('Group: {}'.format(g))
                print('---')
                for d in group.keys():
                    print(group[d])
    return dsAll

def convert_time(delta_time):
    times = []
    years = []
    months = []
    days = []
    hours = []
    minutes = []
    seconds =[]
    
    for i in range(0, len(delta_time)):
        times.append(dt.datetime(1980, 1, 6) + dt.timedelta(seconds = delta_time[i]))
        years.append(times[i].year)
        months.append(times[i].month)
        days.append(times[i].day)
        hours.append(times[i].hour)
        minutes.append(times[i].minute)
        seconds.append(times[i].second)
    
    temp = pd.DataFrame({'time':times, 'year': years, 'month': months, 'day': days,
                         'hour': hours, 'minute': minutes, 'second': seconds
                        })
    return temp

## Read h5 ATL07 files ========================================================
def get_ATL07data(fileT, maxheight, bounding_box, beamlist=None):
    # Pandas/numpy ATL07 reader
        
    f1 = h5py.File(fileT, 'r')

    orient = f1['orbit_info']['sc_orient'][:]  # orientation - 0: backward, 1: forward, 2: transition
    
    if len(orient) > 1:
        print('Transitioning, do not use for science!')
        return [[] for i in beamlist]
    elif (orient == 0):
        beams=['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']                
    elif (orient == 1):
        beams=['gt3r', 'gt3l', 'gt2r', 'gt2l', 'gt1r', 'gt1l']
    # (strong, weak, strong, weak, strong, weak)
    # (beam1, beam2, beam3, beam4, beam5, beam6)

    if beamlist == None:
        beams = [ beams[i] for i in [0, 2, 4]]
    else:
        beams = [ beams[i] for i in beamlist ]
    # use only strong beams

    dL = []

    for beam in beams:
        if beam in list(f1.keys()):
            height=f1[beam]['sea_ice_segments']['heights']['height_segment_height'][:]
            conf=f1[beam]['sea_ice_segments']['heights']['height_segment_confidence'][:]
            rms=f1[beam]['sea_ice_segments']['heights']['height_segment_rms'][:]
            w_gau=f1[beam]['sea_ice_segments']['heights']['height_segment_w_gaussian'][:]
            
            atlas_epoch=f1['/ancillary_data/atlas_sdp_gps_epoch'][0]            
            seg_x = f1[beam]['sea_ice_segments']['seg_dist_x'][:]

            # Delta time in gps seconds
            delta_time = f1[beam]['sea_ice_segments']['delta_time'][:]
            temp = convert_time(delta_time + atlas_epoch)

            # Height segment ID (10 km segments)
            height_segment_id=f1[beam]['sea_ice_segments']['height_segment_id'][:]
            
            mss = f1[beam]['sea_ice_segments']['geophysical']['height_segment_mss'][:]

            lons=f1[beam]['sea_ice_segments']['longitude'][:]
            lats=f1[beam]['sea_ice_segments']['latitude'][:]
            
            seg_len=f1[beam]['sea_ice_segments']['heights']['height_segment_length_seg'][:]
            photon_rate=f1[beam]['sea_ice_segments']['stats']['photon_rate'][:]
            fpb_corr=f1[beam]['sea_ice_segments']['stats']['fpb_corr'][:]
            stdev=f1[beam]['sea_ice_segments']['stats']['height_coarse_stdev'][:]
            
            dF = pd.DataFrame({'beam':beam,
                               'lon':lons, 'lat':lats, 'x': seg_x, 'delta_time':delta_time, 
                               'height_segment_id':height_segment_id, 'height': height, 'conf':conf,
                               'rms': rms, 'w_gau': w_gau, 'seg_len': seg_len, 'photon_rate': photon_rate,
                               'fpb_corr': fpb_corr, 'stdev': stdev, 'mss': mss
                              })
            
            dF = pd.concat([dF, temp], axis=1)
            
        else:
            dF = pd.DataFrame(columns=['beam','lon','lat','delta_time','height_segment_id', 'height', 'conf', 'rms',
                                       'w_gau', 'seg_len', 'photon_rate', 'fpb_corr', 'stdev', 'mss',
                                       'time', 'year', 'month', 'day', 'hour', 'minute', 'second'])

        if len(dF) > 0:
            dF = dF[(dF['height']<maxheight)]
            dF = dF[(dF['lat']>=bounding_box[1])]
            dF = dF[(dF['lat']<=bounding_box[3])]
            dF = dF[(dF['lon']>=bounding_box[0]) | (dF['lon']<=bounding_box[2])]
#             dF = dF[(dF['lon']>=bounding_box[0])]
#             dF = dF[(dF['lon']<=bounding_box[2])]

            # Reset row indexing
            dF=dF.reset_index(drop=True)

            dL.append(dF)
        else:
            dL.append([])              
        
    return dL

## Read h5 ATL10 files ========================================================
def get_ATL10data_extensive(fileT, maxFreeboard, bounding_box, beamlist=[0, 2, 4]):
    """ Pandas/numpy ATL10 reader
    Written by Alek Petty, June 2018 (alek.a.petty@nasa.gov)
    Editted by YoungHyun Ko, Aug 2019 (kooala317@gmail.com)
    
    Args:
        fileT (str): File path of the ATL10 dataset
        maxFreeboard (float): maximum freeboard (meters)
        bounding_box (list): [West bound, South bound, East bound, North bound] 

    returns:
        list of pandas dataframe (6 beams)

    """
    
    #try:
        
    f1 = h5py.File(fileT, 'r')

    orient = f1['orbit_info']['sc_orient'][:]  # orientation - 0: backward, 1: forward, 2: transition
    
    if len(orient) > 1:
        print('Transitioning, do not use for science!')
        return [[] for i in beamlist]
    elif (orient == 0):
        beams=['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']                
    elif (orient == 1):
        beams=['gt3r', 'gt3l', 'gt2r', 'gt2l', 'gt1r', 'gt1l']
    # (strong, weak, strong, weak, strong, weak)
    # (beam1, beam2, beam3, beam4, beam5, beam6)

    beams = [ beams[i] for i in beamlist ]
    # use only strong beams
    
    first = True
    
    for n, beam in enumerate(beams):
        
        if beam in list(f1.keys()):

            freeboard=f1[beam]['freeboard_segment']['beam_fb_height'][:]

            freeboard_confidence=f1[beam]['freeboard_segment']['beam_fb_confidence'][:]
            freeboard_quality=f1[beam]['freeboard_segment']['beam_fb_quality_flag'][:]

            # Delta time in gps seconds
            delta_time = f1[beam]['freeboard_segment']['delta_time'][:]
            # Height segment ID (10 km segments)
            height_segment_id=f1[beam]['freeboard_segment']['height_segment_id'][:]
            
            seg_x=f1[beam]['freeboard_segment']['seg_dist_x'][:] # x-segment distance
            lons=f1[beam]['freeboard_segment']['longitude'][:]
            lats=f1[beam]['freeboard_segment']['latitude'][:]
            deltaTimeRel=delta_time-delta_time[0]

            # #Add this value to delta time parameters to compute full gps_seconds
            atlas_epoch=f1['/ancillary_data/atlas_sdp_gps_epoch'][:] 

            leapSecondsOffset=37
            gps_seconds = atlas_epoch[0] + delta_time - leapSecondsOffset
            # Use astropy to convert from gps time to datetime
            tgps = Time(gps_seconds, format='gps')
            tiso = Time(tgps, format='datetime')
            tiso
            # Conversion of delta_time to a calendar date
            temp = ut.convert_GPS_time(atlas_epoch[0] + delta_time, OFFSET=0.0) 
            
            # sea ice concentration
            sic = f1[beam]['freeboard_segment']['heights']['ice_conc_amsr2'][:] 
            surface_type = f1[beam]['freeboard_segment']['heights']['height_segment_type'][:]
            # resources: ATL07/ATL10 document p. 57 
            # 0: cloud covered, 1: non-lead snow/ice surface
            # 2 & 3: specular lead (low) w/bkg and wo/bkg, respectively
            # 4 & 5: specular lead (high) w/bkg and wo/bkg, respectively
            # 6 & 7: dark lead (smooth) w/bkg and wo/bkg, respectively
            # 8 & 9: dark lead (rough) w/bkg and wo/bkg, respectively
            
            refsur_ndx=f1[beam]['freeboard_segment']['beam_refsurf_ndx'][:]-1
            
            lead = np.zeros(len(refsur_ndx)) 
            
            h5_keys = [key for key in f1[beam].keys()]
            
            if 'leads' in h5_keys:
                lead_ndx = f1[beam]['leads']['ssh_ndx'][:]-1
                lead_n = f1[beam]['leads']['ssh_n'][:]

                for k in range(0, len(lead_ndx)):
                    first_ndx = lead_ndx[k]
                    lead[first_ndx:first_ndx+lead_n[k]] = 1

            year = temp['year'][:].astype('int')
            month = temp['month'][:].astype('int')
            day = temp['day'][:].astype('int')
            hour = temp['hour'][:].astype('int')
            minute = temp['minute'][:].astype('int')
            second = temp['second'][:].astype('int')
            dFtime=pd.DataFrame({'year':year, 'month':month, 'day':day, 
                                'hour':hour, 'minute':minute, 'second':second})
            
            fb_mode = np.zeros(np.shape(freeboard))
            ridge = np.zeros(np.shape(freeboard))
            sample_ndx = np.zeros(np.shape(freeboard))
            
            for ndx in np.unique(refsur_ndx):
                part = (refsur_ndx == ndx)
                x_min = np.min(seg_x[part])
                x_max = np.max(seg_x[part])
                sample_ndx[part] = ndx*10 + (seg_x[part] - x_min)//(10000/3)
                
            for k in np.unique(sample_ndx):                
                subndx = (sample_ndx == k)
                subdata = freeboard[subndx]
                fb_mode[subndx] = calculate_mode(subdata)
            
            fb_mode
            ridge[freeboard > fb_mode + 0.6] = 1            

            dF = pd.DataFrame({'freeboard':freeboard, 'beam':beam, 'beam_num': beamlist[n],
                               'lon':lons, 'lat':lats, 'seg_x': seg_x, 'fb_mode': fb_mode, 'ridge': ridge,
                               'sic': sic, 'stype': surface_type, 'lead': lead, 'refsur_ndx': refsur_ndx,
                               'delta_time':delta_time, 'deltaTimeRel':deltaTimeRel, 
                               'height_segment_id':height_segment_id, 'time': tiso,
                               'year':year, 'month':month, 'day':day, 'hour':hour, 'minute':minute, 'second':second
                              })
                    
        else:
            dF = pd.DataFrame(columns=['freeboard','beam', 'beam_num',
                                       'lon','lat', 'seg_x', 'fb_mode', 'ridge',
                                       'sic', 'stype', 'lead', 'refsur_ndx',
                                       'delta_time','deltiTimeRel',
                                       'height_segment_id', 'datetime', 'year', 'month', 'day',
                                       'hour', 'minute', 'second'])

        if len(dF) > 0:

            #dF['months'] = pd.Series(months, index=dF.index)
            dF = dF[(dF['freeboard']>0)]
            dF = dF[(dF['freeboard']<maxFreeboard)]
            dF = dF[(dF['lat']>=bounding_box[1])]
            dF = dF[(dF['lat']<=bounding_box[3])]
            if bounding_box[0] < bounding_box[2]:
                dF = dF[(dF['lon']>=bounding_box[0])]
                dF = dF[(dF['lon']<=bounding_box[2])]
            else:
                dF = dF[(dF['lon']>=bounding_box[0]) | (dF['lon']<=bounding_box[2])]

            # Reset row indexing
            dF=dF.reset_index(drop=True)
            dF['filename'] = os.path.basename(fileT)
#             dF['fb_mode'] = smooth_line(dF['fb_mode'].values, dF['seg_x'].values, w = 5000)
            
        if first:
            dL = dF
            first = False
        else:
            # dL = dL.append(dF).reset_index(drop=True)
            dL = pd.concat([dL, dF], ignore_index=True)
        
    return dL


## Read h5 ATL10 files ========================================================
def get_ATL10data(fileT, maxFreeboard, bounding_box, beamlist=[0, 2, 4]):
    """ Pandas/numpy ATL10 reader
    Written by Alek Petty, June 2018 (alek.a.petty@nasa.gov)
    Editted by YoungHyun Ko, Aug 2019 (kooala317@gmail.com)
    
    Args:
        fileT (str): File path of the ATL10 dataset
        maxFreeboard (float): maximum freeboard (meters)
        bounding_box (list): [West bound, South bound, East bound, North bound] 

    returns:
        list of pandas dataframe (6 beams)

    """
    
    dL = pd.DataFrame()

    with h5py.File(fileT,'r') as f:

        orient = f['orbit_info']['sc_orient'][:]  # orientation - 0: backward, 1: forward, 2: transition

        if len(orient) > 1:
            print('Transitioning, do not use for science!')
            return [[] for i in beamlist]
        elif (orient == 0):
            beams=['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']                
        elif (orient == 1):
            beams=['gt3r', 'gt3l', 'gt2r', 'gt2l', 'gt1r', 'gt1l']
        # (strong, weak, strong, weak, strong, weak)
        # (beam1, beam2, beam3, beam4, beam5, beam6)

        beams = [ beams[i] for i in beamlist ]
        # use only strong beams

        first = True

        for n, beam in enumerate(beams):

            if beam in list(f.keys()):

                lat = f[beam]['freeboard_segment/latitude'][:]
                lon = f[beam]['freeboard_segment/longitude'][:]
                fb = f[beam]['freeboard_segment/beam_fb_height'][:]

                seg_x = f[beam]['freeboard_segment/seg_dist_x'][:] # (m to km)
                seg_len = f[beam]['freeboard_segment/heights/height_segment_length_seg'][:]
                fb[fb > 100] = np.nan
                stype = f[beam]['freeboard_segment/heights/height_segment_type'][:]
                refsur_ndx = f[beam]['freeboard_segment/beam_refsurf_ndx'][:]
                # Calculate modal freeboard
                freeboard_mode = np.zeros(np.shape(fb))
                ridge = np.zeros(np.shape(fb))
                sample_ndx = np.zeros(np.shape(fb))

                for i in np.unique(refsur_ndx):
                    part = (refsur_ndx == i)
                    x_min = np.min(seg_x[part])
                    x_max = np.max(seg_x[part])
                    sample_ndx[part] = i*10 + (seg_x[part] - x_min)//(10000/3)

                for i in np.unique(sample_ndx):                
                    subndx = (sample_ndx == i)
                    subdata = fb[subndx]
                    freeboard_mode[subndx] = calculate_mode(subdata)

                # Ridge or not? (threshold 0.6 m above level (mode) freeboard)
                ridge[fb > freeboard_mode + 0.6] = 1
                
                lead_mask = np.zeros(len(fb))
                lead_mask[(stype <= 5) & (stype>= 2) & (fb < 0.1)] = 1 # Specular leads

                dF = pd.DataFrame({'lat': lat, 'lon': lon, 'seg_x': seg_x, 'seg_len': seg_len, 'sample': sample_ndx,
                                   'freeboard': fb, 'fb_mode': freeboard_mode, 'ridge': ridge, 'stype': stype, 'lead': lead_mask})
                dF['beam'] = beam

                if len(dF) > 0:

                    #dF['months'] = pd.Series(months, index=dF.index)
                    dF = dF[(dF['freeboard']>0)]
                    dF = dF[(dF['freeboard']<maxFreeboard)]
                    dF = dF[(dF['lat']>=bounding_box[1])]
                    dF = dF[(dF['lat']<=bounding_box[3])]
                    if bounding_box[0] < bounding_box[2]:
                        dF = dF[(dF['lon']>=bounding_box[0])]
                        dF = dF[(dF['lon']<=bounding_box[2])]
                    else:
                        dF = dF[(dF['lon']>=bounding_box[0]) | (dF['lon']<=bounding_box[2])]

                    # Reset row indexing
                    dF=dF.reset_index(drop=True)
                    dF['filename'] = os.path.basename(fileT)

                if first:
                    dL = dF
                    first = False
                else:
                    # dL = dL.append(dF).reset_index(drop=True)
                    dL = pd.concat([dL, dF], ignore_index=True)
        
    return dL


# Calculate modal freeboard based on freeboard distribution
def calculate_mode(data, N = 20):
    data = data[~np.isnan(data)]
    w = 0.02
    M = 4
    m = w    
    
    if len(data) > N: # minimum number of freeboard observations to create distribution
        # instantiate and fit the KDE model
        kde = KernelDensity(bandwidth=w*5, kernel='gaussian')
        kde.fit(data[:, None])
        x_d = np.arange(m, M, w)
        logprob = kde.score_samples(x_d[:, None])
        n_max = np.argmax(np.exp(logprob))
        mode = x_d[n_max]
    # elif len(data) > 0: # if the number of observation is not enough, just take 0.4 quantile
    #     mode = np.quantile(data, 0.4)
    else:
        mode = np.nan

    return mode



def smooth_line(data, x, w = 2):
    # Smooth the surface with the defined window size
    output = np.zeros(len(data))
    for n in range(0, len(data)):
#         output[n] = np.mean(data[max(0, n-w):min(len(data), n+w+1)])
        output[n] = np.mean(data[(x <= x[n]+w)&(x >= x[n]-w)])
    return output

def calculate_ridge_old(freeboard):
    
    mode, ridge_fr, ridge_h = 0,0,0;
    
    if len(freeboard) > 0:
        mode = calculate_mode(freeboard)
        ridge_fr = len(freeboard[freeboard > mode+0.6])/len(freeboard)*100
        ridge_h = np.nanmean(freeboard[freeboard > mode+0.6]) - mode
    
    return [mode, ridge_fr, ridge_h]

# Calculate ridge fraction etc.
def calculate_ridge(df):
    
    mode, ridge_fr, ridge_h = 0,0,0;
    std, mean, med = 0,0,0;
    diff = df['freeboard'].values - df['fb_mode'].values
    
    if len(df) > 0:
        
        mode = np.nanmean(df['fb_mode'].values)
        std = np.nanstd(df['freeboard'].values)
        mean = np.nanmean(df['freeboard'].values)
        med = np.nanmedian(df['freeboard'].values)
        ridge_h = np.nanmean(diff[df['ridge']==1])
        ridge_fr = len(diff[df['ridge']==1]) / len(diff) * 100
        
#         mode = calculate_mode(freeboard)
#         ridge_fr = len(freeboard[freeboard > mode+0.6])/len(freeboard)*100
#         ridge_h = np.nanmean(freeboard[freeboard > mode+0.6]) - mode
    
    return [mode, ridge_fr, ridge_h, mean, med, std]

def calculate_lead(lead):
    return len(lead[lead == 1])/len(lead)*100


def get_ATL10lead(fileT, maxFreeboard, bounding_box, beamlist=None):
    # Pandas/numpy ATL10 reader
        
    f1 = h5py.File(fileT, 'r')

    orient = f1['orbit_info']['sc_orient'][:]  # orientation - 0: backward, 1: forward, 2: transition
    
    if len(orient) > 1:
        print('Transitioning, do not use for science!')
        return [[] for i in beamlist]
    elif (orient == 0):
        beams=['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']                
    elif (orient == 1):
        beams=['gt3r', 'gt3l', 'gt2r', 'gt2l', 'gt1r', 'gt1l']
    # (strong, weak, strong, weak, strong, weak)
    # (beam1, beam2, beam3, beam4, beam5, beam6)

    if beamlist == None:
        beams = [ beams[i] for i in [0, 2, 4]]
    else:
        beams = [ beams[i] for i in beamlist ]
    # use only strong beams

    dL = []

    for beam in beams:
        if beam in list(f1.keys()):
            lead_height=f1[beam]['leads']['lead_height'][:]
            lead_length=f1[beam]['leads']['lead_length'][:]
            lead_sigma=f1[beam]['leads']['lead_sigma'][:]

            atlas_epoch=f1['/ancillary_data/atlas_sdp_gps_epoch'][0]

            # Delta time in gps seconds
            delta_time = f1[beam]['leads']['delta_time'][:]
            temp = convert_time(delta_time + atlas_epoch)

            # Height segment ID (10 km segments)
            ssh_n=f1[beam]['leads']['ssh_n'][:]
            ssh_ndx=f1[beam]['leads']['ssh_ndx'][:]

            lons=f1[beam]['leads']['longitude'][:]
            lats=f1[beam]['leads']['latitude'][:]

            dF = pd.DataFrame({'beam':beam, 'height':lead_height, 'length': lead_length, 'sigma': lead_sigma,
                               'lon':lons, 'lat':lats, 'delta_time':delta_time, 
                               'ssh_n':ssh_n, 'ssh_ndx': ssh_ndx
                              })
            
            dF = pd.concat([dF, temp], axis=1)
            
        else:
            dF = pd.DataFrame(columns=['beam', 'height','length', 'sigma', 'delta_time',
                            'ssh_n', 'ssh_ndx', 'year', 'month', 'day',
                            'hour', 'minute', 'second'])

        if len(dF) > 0:

            dF = dF[(dF['lat']>=bounding_box[1])]
            dF = dF[(dF['lat']<=bounding_box[3])]
            # dF = dF[(dF['lon']>=bounding_box[0]) | (dF['lon']<=bounding_box[2])]
            dF = dF[(dF['lon']>=bounding_box[0])]
            dF = dF[(dF['lon']<=bounding_box[2])]

            # Reset row indexing
            dF=dF.reset_index(drop=True)

            dL.append(dF)
        else:
            dL.append([])              
        
    return dL


def MakeDataSet(LocalFilePath, beam='gt1r'):
    # This code written by Cecilia Bitz  reads in data from an ATL07 dataset and then spits out an xarray
    
    ATL07 = h5py.File(LocalFilePath, 'r')

    # coordinates, start their lives as data arrays
    lons = xr.DataArray(ATL07[beam+'/sea_ice_segments/longitude'][:],dims=['segs'])
    lons.name='lons'
    lats = xr.DataArray(ATL07[beam+'/sea_ice_segments/latitude'][:],dims=['segs'])
    lats.name='lats'
    # add 360 to lons less than 0
    lons360 = lons.where(lons.values>0, other=lons.values+360)

    # this is the time hacked a bit since I am an idiot, it is within seconds
    delta_time=ATL07[beam+'/sea_ice_segments/delta_time'][:] 
    time = np.datetime64('2018-01-01') + (delta_time-86400*0.015).astype('timedelta64[s]' ) 

    # variables in datasets, start their lives as data arrays too
    seg_dist = xr.DataArray(ATL07[beam+'/sea_ice_segments/seg_dist_x'][:],dims=['segs'])
    seg_dist.name = 'seg_dist' # the first two dataarrays have to be named, grr
    #print(set_dist)

    height = xr.DataArray(ATL07[beam+'/sea_ice_segments/heights/height_segment_height'][:],dims=['segs'])
    height.name = 'height'
    #print('\n\nTake a look at the dataarray we made \n')
    #print(height)
    
    mss = xr.DataArray(ATL07[beam+'/sea_ice_segments/geophysical/height_segment_mss'][:],dims=['segs'])
    seg_length= xr.DataArray(ATL07[beam+'/sea_ice_segments/heights/height_segment_length_seg'][:],dims=['segs'])
    quality_flag = xr.DataArray(ATL07[beam+'/sea_ice_segments/heights/height_segment_fit_quality_flag'][:],dims=['segs'])
    isita_lead =   xr.DataArray(ATL07[beam+'/sea_ice_segments/heights/height_segment_ssh_flag'][:],dims=['segs'])

    # start by merging first two datarrays (they have to have names)
    ds=xr.merge([seg_dist, height])

    # now we add more dataarrays 
    ds['mss'] = mss
    ds['seg_length'] = seg_length
    ds['quality_flag'] = quality_flag
    ds['isita_lead'] = isita_lead
    
    ds.coords['lon'] = lons
    ds.coords['lat'] = lats
    ds.coords['time'] = xr.DataArray(time,dims=['segs'])
    ds.coords['delta_time'] = xr.DataArray(delta_time,dims=['segs'])

    ds.coords['lon360'] = lons360
    ds.coords['segs'] = xr.DataArray(np.arange(0,len(height),1),dims=['segs'])

#    print('\n\nTake a look at the dataset we made \n')
#    print(ds)
    ATL07 = None

    return ds

def MultiFileDataSet(multiple_files, beams):
    # This code written by Cecilia Bitz reads in data from an ATL07 dataset and then spits out an xarray
    
    num_beams = np.size(beams) # do not use len !
    num_files = np.size(multiple_files)
    ds_beams = None
    ds_all = None
    file = None
    beam = None

    if (num_files>1):
        ifile = 0
        for file in multiple_files:
            if num_beams>1:
                ibeams=0
                for beam in beams:
                    #print(beam)
                    ds = MakeDataSet(file, beam)
                    if ibeams==0:
                        ds_beams = ds
                    else:
                        ds_beams =xr.concat([ds, ds_beams])

                    ibeams = ibeams+1
                ds_beams = ds_beams.rename({'concat_dims':'beam'})
            else:
                ds_beams = MakeDataSet(file, beams[0])

            if ifile==0:
                ds_all = ds_beams
            else:
                ds_all =xr.concat([ds_beams, ds_all])
            ifile=ifile+1

    #    print(ds_all)
        ds_all = ds_all.rename({'concat_dims':'track'})
        ds_all.coords['track'] = xr.DataArray(np.arange(0,num_files,1),dims=['track'])
        if num_beams > 1:
            ds_all = ds_all.transpose('segs','track','beam')
            ds_all.coords['beam'] = xr.DataArray(np.arange(0,num_beams,1),dims=['beam'])

    else:
        # just one file
        print('one file')
        if (num_beams>1):
            ibeams=0
            for beam in beams:
                #print(beam)
                ds = MakeDataSet(multiple_files[0], beam)
                if ibeams==0:
                    ds_beams = ds
                else:
                    ds_beams =xr.concat([ds, ds_beams])

                ibeams = ibeams+1
            ds_all = ds_beams.rename({'concat_dims':'beam'})
            ds_all.coords['beam'] = xr.DataArray(np.arange(0,num_beams,1),dims=['beam'])
        else:
            ds_all = MakeDataSet(multiple_files[0], beams[0])

    ds=None
    ds_beams = None

    return ds_all

from scipy.interpolate import griddata

def snow_depth(fb, xx, yy, date, method = "KK"):
    # fb: freeboard (total snow freeboard; units in meters)
    if method == "KK": # Kacimi & Kwok, 2020 (ICESat-2, CryoSat-2)
        if date.month == 4:
            hs = fb*0.69 - 0.0397        
        elif date.month == 5:
            hs = fb*0.68 - 0.0372
        elif date.month == 6:
            hs = fb*0.69 - 0.0397
        elif date.month == 7:
            hs = fb*0.69 - 0.0397
        elif date.month == 8:
            hs = fb*0.69 - 0.0410            
        elif date.month == 9:
            hs = fb*0.66 - 0.0420
        elif date.month == 10:
            hs = fb*0.66 - 0.0420 
        else:          
            hs = fb*0.66 - 0.0345            
            
    elif method == "AMSR": # Snow depth from AMSR data
        hs = read_AMSR(date, xx, yy)
        
    elif method == "zero": # Zero ice freeboard
        hs = fb
        
    elif method == "XOC": # Ozsoy-Cicek et al., 2013 (Empirical equation)
        hs = 1.05*fb - 0.005
            
    return hs

def ice_thickness(fb, hs):
    pw = 1024
    ps = 340
    pi = 917
    hi = (pw/(pw-pi))*fb + ((ps-pw)/(pw-pi))*hs
    return hi

def read_AMSR(date, xx, yy):

    h5file = "D:\\Ross\\AMSR\\AMSR_U2_L3_SeaIce12km_B04_{0}.he5".format(dt.datetime.strftime(date, "%Y%m%d"))
    
    with h5py.File(h5file) as f:
        lat2 = f['HDFEOS']['GRIDS']['SpPolarGrid12km']['lat'][:]
        lon2 = f['HDFEOS']['GRIDS']['SpPolarGrid12km']['lon'][:]
        sd = f['HDFEOS/GRIDS/SpPolarGrid12km/Data Fields/SI_12km_SH_SNOWDEPTH_5DAY'][:] * 0.01  # cm to m
        
        sd[sd > 0.7] = np.nan

        # EPSG:4326 (WGS84); EPSG:3408 (NSIDC EASE-Grid North - Polar pathfinder sea ice movement)
        inProj = Proj('epsg:4326')  
        outProj = Proj('epsg:3412')
        xx2,yy2 = transform(inProj,outProj,lat2,lon2)
        grid_sd = griddata((xx2.flatten(), yy2.flatten()), sd.flatten(), (xx, yy), method='nearest')
    
    return grid_sd 
