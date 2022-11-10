import numpy as np
import metpy
import metpy.calc as mpcalc
def log_r2( modelled_weathering, obs_weathering):

    RSS = sum((np.log(modelled_weathering) - np.log(obs_weathering))**2)
    TSS = sum((np.log(obs_weathering) - np.mean(np.log(obs_weathering)))**2)
    
    r2 = 1 - RSS/TSS
    
    return r2

def get_basin_weathering(variables, forcings):
    '''
    constants are:
    1/ suture map
    2/ tslope map
    3/ T map
    4/ Q map
    5/ polygon area of river basins
    '''
    tslope = forcings[0]
    T = forcings[1]
    Q = forcings[2]
    polygon_area = forcings[3]
    SUTURE = forcings[4]
    ARC = forcings[5]
    RELICT = forcings[6]
    ROOTS = forcings[7]
    
    ##### Maffre erosion calculation
    k_erosion = 6.5e-4 #### use normal value
    #### Maffre params
    Xm = 0.1 #cation abundance in source rock
    Ea = 20 #apparent activation energy
    z = 10 #regolith (kg/m2)
    #callibration constants
    K = 6.2e-5 #variables[0]
    kw = 8.14e-1 #variables[1]
    sigplus1 = 0.636 #variables[2]*1000#
    arc_factor = 12#variables[3]*1000000
    suture_factor = 70#variables[4]*1000000
    relict_factor = 12#arc_factor
    root_factor = variables[0]*1000000

    print(K, kw, sigplus1, arc_factor, suture_factor, root_factor)
    
    ##### fixed params
    T0 = 286 #standard temp in K
    R = 8.31e-3 #ideal gas constant

    TC = T - 273
    epsilon = k_erosion * (Q**0.31) * tslope * np.maximum(TC, 2)

    #### T, Q and erosion dependencies (but don't matter since we're taking directly from model results)
    R_T = np.exp( ( Ea / (R*T0) ) - ( Ea / (R*T) ) )
    R_Q = 1 - np.exp( -1 * kw * Q )
    R_reg = ( (z/epsilon)**sigplus1) / sigplus1

    #mask out some areas
    arc_mask = ARC != 0
    suture_mask = SUTURE != 0
    relict_mask = RELICT != 0
    root_mask = ROOTS != 0

    
    #### West (2012) combined dependency model, *1e6 to get to t/km2/yr
    CW_raw = epsilon * Xm * ( 1 - np.exp( -1* K * R_Q * R_T * R_reg ) ) * 1e6
    CW_raw_AF = CW_raw *( 1 + ARC * ( arc_factor - 1 ) ) * arc_mask
    CW_raw_SF = CW_raw *( 1 + SUTURE * ( suture_factor - 1 ) ) * suture_mask
    CW_raw_RAF = CW_raw *( 1 + RELICT * ( relict_factor - 1 ) ) * relict_mask
    CW_raw_ROOTS = CW_raw *( 1 + ROOTS * ( root_factor - 1 ) ) * root_mask
    non_arc_suture_weathering_present = CW_raw * (arc_mask != True) * (suture_mask != True) \
                                      * (relict_mask != True) * (root_mask != True)

    CW = CW_raw_AF + CW_raw_SF + CW_raw_RAF + CW_raw_ROOTS + non_arc_suture_weathering_present
    
    bulkbasinweathering = sum_basin_weathering_FOAM(polygon_area, CW)
    
    return bulkbasinweathering, CW_raw_AF, CW_raw_SF, CW_raw_RAF, CW_raw_ROOTS, CW

def sum_basin_weathering_PD(polygon_area, CW):
    
    '''
    Sum weathering in each given polygon using present-day (empirical) data
    '''

    bulkbasinweathering = np.zeros([len(polygon_area),])

    for i in np.arange(len(polygon_area)):

        thisarea = np.flipud(polygon_area[i,:,:]/1000000)
        #check this line
        basinweathering = CW*thisarea.data
        #print(basinweathering)
        basinweathering[np.isnan(basinweathering)] = 0
        bulkbasinweathering[i] = np.nansum(np.nansum(basinweathering))

    return bulkbasinweathering

def sum_basin_weathering_FOAM(polygon_area, CW):
     
    '''
    Sum weathering in each given polygon using FOAM modelled data data
    '''

    bulkbasinweathering = np.zeros([len(polygon_area),])

    for i in np.arange(len(polygon_area)):

        thisarea = polygon_area[i,:,:]/1000000
        #check this line
        basinweathering = CW*np.roll(thisarea.data, 24, axis=1)
        #print(basinweathering)
        basinweathering[np.isnan(basinweathering)] = 0
        bulkbasinweathering[i] = np.nansum(np.nansum(basinweathering))

    return bulkbasinweathering

def get_HADCM3_weathering_PD(tmp_avg, topo, lat, lon, run_data, polygon_area_PD):
    ################### HadCM3 present day ######################
    ############################################################

    #### weathering calculation
    T = tmp_avg + 273 ### K
    # Q = pre_avg ### mm/yr
    Q = run_data ### mm/yr
    height = topo ### m

    #gradient of slope
    y_lat = np.asarray(lat.reshape(360,))
    x_lon = np.asarray(lon.reshape(720,))
    #get meshgrid
    d_lon, d_lat = metpy.calc.lat_lon_grid_deltas(np.asarray(x_lon), np.asarray(y_lat))
    #calculate gradient
    dFdyNorth, dFdxEast = mpcalc.gradient(height, deltas=(d_lat, d_lon))
    #calculate slope
    tslope = np.asarray(( dFdyNorth**2 + dFdxEast**2 )**0.5)

    ##### pierre erosion calculation
    k_erosion = 6.5e-4
    TC = tmp_avg
    epsilon = k_erosion * (Q**0.31) * tslope *  np.maximum(TC,2)

    #### Pierre params
    Xm = 0.1
    K = 6e-5
    kw = 1e-3
    Ea = 20
    z = 10
    sigplus1 = 0.9

    ##### fixed params
    T0 = 286
    R = 8.31e-3

    ##### T, Q and erosion dependencies
    R_T = np.exp( ( Ea / (R*T0) ) - ( Ea / (R*T) ) )
    R_Q = 1 - np.exp( -1*kw * Q )
    R_reg = ( (z/epsilon)**sigplus1 ) / sigplus1

    # West (2012) combined dependency model, *1e6 to get to t/km2/yr
    CW = epsilon * Xm * ( 1 - np.exp( -1 * K * R_Q * R_T * R_reg ) ) * 1e6

    return sum_basin_weathering_PD(polygon_area_PD, CW)    