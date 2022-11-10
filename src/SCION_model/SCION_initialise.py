import numpy as np
import time
import SCION_classes
import SCION_equations
import SCION_plot_fluxes
import SCION_plot_worldgraphic
import pandas as pd
import scipy.io as sp
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def intersect_mtlb(a, b):
    #same functionality as intersect in matlab
    #taken from:
    #https://stackoverflow.com/questions/45637778/how-to-find-intersect-indexes-and-values-in-python
    #
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]

def SCION_initialise(S):
    start_time = time.time()
    ######################################################################
    #################   Check for sensitivity analysis   #################
    ######################################################################
    if S >= 1:
        #for sensitivity analysis
        sensanal = SCION_classes.Sensanal_class(sensanal_key = 1)
        plotrun = SCION_classes.Plotrun_class(plotrun_key = 0)
        telltime = 0
    else:
        #for other runs
        sensanal = SCION_classes.Sensanal_class(sensanal_key = 0)
        plotrun = SCION_classes.Plotrun_class(plotrun_key = 1)
        telltime = 1
    #set Gtune option
    #default is off
    Gtune = SCION_classes.Gtune_class(gtune_key=0)
    ####### starting to load params
    if sensanal.key == 0:
        print('setting parameters... %s' % (time.time() - start_time))

    #set flags for testing drivers
    #0 =default,  driver is on
    #1 = driver is off (i.e equal to present-day, or 1)
    #this is probably a confusing way to do it, future step to change
    #and introduce a keyword for testing automatically
    BIO_TEST = 0
    ARC_TEST = 0
    SUTURE_TEST = 0
    PALAEOGEOG_TEST = 0
    DEGASSING_TEST = 0

    if ARC_TEST == 1:
        print('Arcs are turned OFF!')
    else:
        print('Arcs are turned ON!')
    if SUTURE_TEST == 1:
        print('Sutures are turned OFF')
    else:
        print('Sutures are turned ON!')
    if DEGASSING_TEST == 1:
        print('Degassing is turned OFF!')
    else:
        print('Degassing is turned ON!')
    if BIO_TEST == 1:
        print('Bio is turned OFF!')
    else:
        print('Bio is turned ON!')
    if PALAEOGEOG_TEST == 1:
        print('PALAEOGEOG is turned OFF!')
    else:
        print('PALAEOGEOG is turned ON!')
    ####################################################################
    ####################   Flux values at present   ####################
    ####################################################################

    #### reductant input
    k_reductant_input = 0.4e12 ### schopf and klein 1992

    #### org C cycle
    k_locb = 2.5e12
    k_mocb = 2.5e12
    k_ocdeg = 1.25e12

    #### carb C cycle
    k_ccdeg = 12e12
    k_carbw = 8e12
    k_sfw = 1.75e12
    basfrac = 0.3

    #### S cycle
    k_mpsb = 0.7e12
    k_mgsb = 1e12
    k_pyrw = 7e11
    k_gypw = 1e12
    k_pyrdeg = 0
    k_gypdeg = 0

    #### P cycle
    k_capb = 2e10
    k_fepb = 1e10
    k_mopb = 1e10
    k_phosw = 4.25e10
    k_landfrac = 0.0588
    #### N cycle
    k_nfix = 8.67e12
    k_denit = 4.3e12

    #### Sr cycle
    k_Sr_sedw = 17e9
    k_Sr_mantle = 7.3e9
    k_Sr_silw = 13e9
    k_Sr_metam = 13e9

    #### others
    k_oxfrac = 0.9975
    Pconc0 = 2.2
    Nconc0 = 30.9
    newp0 = 117 * min(Nconc0/16,Pconc0)
    #COPSE constant for calculating pO2 from normalised O2
    copsek16 = 3.762
    #oxidative weathering dependency on O2 concentration
    a = 0.5
    #marine organic carbon burial dependency on new production
    b = 2
    ##fire feedback
    kfire= 3

    #reservoir present day sizes (mol)
    P0 = 3.1*10**15
    O0 = 3.7*10**19
    A0 = 3.193*10**18
    G0 = 1.25*10**21
    C0 = 5*10**21
    PYR0 = 1.8*10**20
    GYP0 = 2*10**20
    S0 = 4*10**19
    CAL0 = 1.397e19
    N0 = 4.35e16
    OSr0 = 1.2e17 ### francois and walker 1992
    SSr0 = 5e18

    #arc and suture enhancement factor
    arc_factor = 12#7#10
    suture_factor = 80#343
    relict_arc_factor = 12#7#10

    #change in root depths
    #root_times = np.asarray([-600, ,-410, -400, -350, 0])#default
    root_times = np.asarray([-600, -410, -400, -350, 0])
    #460 from Canadell et al. 1996 Oecologia, expressed as a fraction out of 1 (== present day)
    max_root_enhancement_factor = 1.3
    root_depth_raw = np.asarray([0, 0, 10, 100, 460])
    root_depths = root_depth_raw*max_root_enhancement_factor/root_depth_raw[-1] #default [0, 0, 10, 100, 460]
    #root_depths = np.asarray([0, 0, 10, 200, 460])/460
    #root_depths = np.asarray([1, 1, 1, 1, 1])*max_root_enhancement_factor/1 
    #root_depths = np.asarray([0, 0, 0, 0, 0])/1
    #print(root_depths)
    #root_interp = interp1d(root_times, root_depths)
    root_depth_factor = interp1d(root_times, root_depths)

    #if testing them
    if ARC_TEST == 1:
        arc_factor = 1
        relict_arc_factor = 1
    if SUTURE_TEST == 1:
        suture_factor = 1

    #### finished loading params
    if sensanal.key == 0:
        print('Done')
        endtime = time.time() - start_time
        print('time:', endtime)

    #####################################################################
    #################   Load Forcings   #################################
    #####################################################################

    ####### starting to load forcings
    if sensanal.key == 0:
        print('loading forcings... %s' % (time.time() - start_time))

    ### load INTERPSTACK
    file_to_open_INTERPSTACK ='./forcings/INTERPSTACK_sep2021_v5.mat'
    mat_contents = sp.loadmat(file_to_open_INTERPSTACK)

    CO2 = mat_contents['INTERPSTACK'][0][0][0][0]
    interp_time = mat_contents['INTERPSTACK'][0][0][1][0]
    Tair = mat_contents['INTERPSTACK'][0][0][2]
    runoff = mat_contents['INTERPSTACK'][0][0][3]
    land = mat_contents['INTERPSTACK'][0][0][4]
    lat = mat_contents['INTERPSTACK'][0][0][5][0]
    lon = mat_contents['INTERPSTACK'][0][0][6][0]
    topo = mat_contents['INTERPSTACK'][0][0][7]
    aire = mat_contents['INTERPSTACK'][0][0][8]
    gridarea = mat_contents['INTERPSTACK'][0][0][9]
    suture = mat_contents['INTERPSTACK'][0][0][10]
    #arc = mat_contents['INTERPSTACK'][0][0][11]#default arc from Cao
    arc = mat_contents['INTERPSTACK'][0][0][12]#Cao arc-suture (as in, Cao arc subtract suture)
    relict_arc = mat_contents['INTERPSTACK'][0][0][13]
    #arc = mat_contents['INTERPSTACK'][0][0][14]#default plate model arcs
    #arc = mat_contents['INTERPSTACK'][0][0][15]#default plate model arcs-suture
    slope = mat_contents['INTERPSTACK'][0][0][16]
    #make sure normal arcs are gone from relict arcs
    relict_arc = relict_arc*(arc == 0).astype(int)
    #sutures are rastersied at width of ~20 km (18.75 to be precise()), but usually 3 times wider
    suture = np.minimum(1, (suture * 3))
    #subtract sutures from relict arcs
    relict_arc = relict_arc-suture
    relict_arc = np.maximum(0, (relict_arc))
    #### root depths
    file_to_open_ROOTS = './forcings/root_presence.mat'
    root_maps = sp.loadmat(file_to_open_ROOTS)
    root_presence = root_maps['root_presence']
    #Maffre and West params, SCION default are greyed out
    Xm = 0.1
    K =  6.2e-5 #6e-5 | 3.8e-5
    kw = 8.14e-1 # 1e-3 |6.3e-1
    Ea = 20
    z = 10
    sigplus1 = 0.636#0.9
    T0 = 286
    R = 8.31e-3

    erosion_pars = SCION_classes.Erosion_parameters_class(Xm, K, kw, Ea, z,
                                                            sigplus1, T0, R)


    #### load COPSE reloaded forcing set
    file_to_open_FORCINGS =  './forcings/COPSE_forcings_June2022_v2.mat'
    forcings_contents = sp.loadmat(file_to_open_FORCINGS)


    t = forcings_contents['forcings'][0][0][0][0]
    #update this to dietmar's paper
    B = forcings_contents['forcings'][0][0][1][2]
    BA = forcings_contents['forcings'][0][0][2][0]
    Ca = forcings_contents['forcings'][0][0][3][0]
    CP = forcings_contents['forcings'][0][0][4][0]
    D = forcings_contents['forcings'][0][0][5][0]
    E = np.ones_like(forcings_contents['forcings'][0][0][6][0])
    #E = forcings_contents['forcings'][0][0][6][0]
    GA = forcings_contents['forcings'][0][0][7][0]
    PG = forcings_contents['forcings'][0][0][8][0]
    U = forcings_contents['forcings'][0][0][9][0]
    W = np.ones_like(forcings_contents['forcings'][0][0][10][0])
    #W = forcings_contents['forcings'][0][0][10][0]
    coal = forcings_contents['forcings'][0][0][11][0]
    epsilon = forcings_contents['forcings'][0][0][12][0]
    if BIO_TEST == 1:
        W = np.ones_like(forcings_contents['forcings'][0][0][10][0])
        E = np.ones_like(forcings_contents['forcings'][0][0][6][0])
    #### new BA
    file_to_open_GR_BA = './forcings/GR_BA.xlsx'
    GR_BA_df = pd.read_excel(file_to_open_GR_BA)
    #### new GA
    file_to_open_GA = './forcings/GA_revised.xlsx'
    GA_df = pd.read_excel(file_to_open_GA)

    #### degassing rate
    file_to_open_DEGASSING = './forcings/combined_D_force_revised_Oct2022.mat'
    forcing_degassing = sp.loadmat(file_to_open_DEGASSING)
    #### load shoreline forcing
    file_to_open_SHORELINE = './forcings/shoreline.mat'
    forcing_shoreline = sp.loadmat(file_to_open_SHORELINE)

    ####define forcings class here
    forcings = SCION_classes.Forcings_class(t, B, BA, Ca, CP, D, E, GA, PG, U,
                                            W, coal, epsilon, GR_BA_df, GA_df,
                                            forcing_degassing, forcing_shoreline)
    #make interpolations
    forcings.get_interp_forcings()

    #### finished loading forcings
    if sensanal.key == 0:
        print('Done')
        endtime = time.time() - start_time
        print('time:', endtime)
        #dont need sensparams, so we can pass an empty array
        sensparams = np.asarray([])
    #####################################################################
    #################   Generate sensitivity randoms   ##################
    #####################################################################

    if sensanal.key == 1:
        #set the random seed so we get pseudo random each iteration and process
        np.random.seed()
        #### generate random number in [-1 +1]
        sensparams = SCION_classes.Sensparams_class(randminusplus1 = 2*(0.5-np.random.rand(1)),
                                                    randminusplus2 = 2*(0.5-np.random.rand(1)),
                                                    randminusplus3 = 2*(0.5-np.random.rand(1)),
                                                    randminusplus4 = 2*(0.5-np.random.rand(1)),
                                                    randminusplus5 = 2*(0.5-np.random.rand(1)),
                                                    randminusplus6 = 2*(0.5-np.random.rand(1)),
                                                    randminusplus7 = 2*(0.5-np.random.rand(1)))

    #####################################################################
    #######################   Initialise solver   #######################
    #####################################################################

    #### run beginning
    if sensanal.key == 0:
        print('Beginning run: \n')

    #### if no plot or sensitivity command set to single run

    if not sensanal.key:
        sensanal.key = 0
    if sensanal.key != 1:
        if not plotrun.key:
            plotrun.key = 1


    ###model timeframe in years (0 = present day)
    whenstart = -600e6
    whenend = -0

    #### setp up grid stamp times
    gridstamp_number = 0
    finishgrid = 0

    ####### set number of model steps to take before bailing out
    bailnumber = 1e5

    ####### display every n model steps whilst running
    display_resolution = 200
    ####set output length to be 0 for now
    output_length = 0
    ###
    #####define model parameters class here
    ###

    ###define INTERPSTACK here
    INTERPSTACK = SCION_classes.Interpstack_class(CO2, interp_time, Tair, runoff,
                                                  land, lat, lon, topo, aire,
                                                  gridarea, suture, arc, relict_arc,
                                                  slope, root_presence)


    model_pars = SCION_classes.Model_parameters_class(
                    whenstart, whenend, INTERPSTACK.time, gridstamp_number,
                    finishgrid, bailnumber, display_resolution, output_length)

    #### relative contribution from latitude bands
    model_pars.get_rel_contrib(INTERPSTACK.lat, INTERPSTACK.lon)

    #to test palaeogeography
    if PALAEOGEOG_TEST == 1:
    #GAST = np.mean(Tair_past * model_pars.rel_contrib)*contribution_past  +  np.mean(Tair_future * model_pars.rel_contrib)*contribution_future
        for time_step in np.arange(np.shape(Tair)[3]):
            for CO2_step in np.arange(np.shape(Tair)[2]):
                #get mean tair value (note, Tair is a full coverage grid, so don't need to use land to filter 1/0s)
                tmp_tair_mean = np.mean(Tair[:,:,CO2_step,time_step] * model_pars.rel_contrib)
                #get land mask
                tmp_land_tair = np.ones_like(land[:,:,time_step]) * tmp_tair_mean
                Tair[:,:,CO2_step,time_step] = tmp_land_tair

        for time_step in np.arange(np.shape(runoff)[3]):
            for CO2_step in np.arange(np.shape(runoff)[2]):
                #get mean runoff value
                tmp_runoff_mean = runoff[:,:,CO2_step,time_step][np.nonzero(runoff[:,:,CO2_step,time_step])].mean()
                #these lines ensure we don't have 0 values in our final calc, see CO2_vs_CWeathering_tot.ipynb
                if tmp_runoff_mean < 4 :
                    tmp_runoff_mean = 4
                #get land mask
                tmp_land_runoff = np.copy(land[:,:,time_step]) * tmp_runoff_mean * 0.25
                runoff[:,:,CO2_step,time_step] = tmp_land_runoff

        for time_step in np.arange(np.shape(slope)[2]):

            tmp_slope_mean = slope[:,:,time_step][np.nonzero(slope[:,:,time_step])].mean()
            tmp_land_slope = np.copy(land[:,:,time_step]) * tmp_slope_mean
            slope[:,:,time_step] = tmp_land_slope

    #### define pars class here###
    pars = SCION_classes.Variable_parameters_class(
            telltime, k_reductant_input, k_mocb, k_locb, k_ocdeg, k_ccdeg,
            k_carbw, k_sfw,basfrac, k_mpsb, k_mgsb, k_pyrw, k_gypw, k_pyrdeg,
            k_gypdeg, k_capb, k_fepb, k_mopb, k_phosw, k_landfrac, k_nfix,
            k_denit, k_Sr_sedw, k_Sr_mantle, k_Sr_silw, k_Sr_metam, k_oxfrac,
            newp0, copsek16, a, b, kfire, P0, O0, A0, G0, C0, PYR0, GYP0, S0,
            CAL0, N0, OSr0, SSr0, suture_factor, arc_factor, relict_arc_factor, root_depth_factor,
            PALAEOGEOG_TEST, BIO_TEST, DEGASSING_TEST, ARC_TEST, SUTURE_TEST)

    #define suture/arc get_masks
    INTERPSTACK.get_masks()
    INTERPSTACK.get_enhancements(pars)

    ###
    #### define stepnumber class
    ###
    step = 1
    stepnumber = SCION_classes.Stepnumber_class(step)

    #### set starting reservoir sizes
    pstart = pars.P0
    tempstart = 288
    CAL_start = pars.CAL0
    N_start = pars.N0
    OSr_start = pars.OSr0
    SSr_start = pars.SSr0
    delta_A_start = 0
    delta_S_start = 35
    delta_G_start = -27
    delta_C_start = -2
    delta_PYR_start = -5
    delta_GYP_start = 20
    delta_OSr_start = 0.708
    delta_SSr_start = 0.708

    #####################################################################
    ################   Initial parameter tuning option  #################
    #####################################################################

    #define tuning options (default is use pretuned)
    #if using new tune values (Gtune.key=1)
    if Gtune.key == 1:
        ostart = pars.O0 * abs( Otune )
        astart = pars.A0 * abs( Atune )
        sstart = pars.S0 * abs( Stune )
        gstart = pars.G0 * abs( Gtune )
        cstart = pars.C0 * abs( Ctune )
        pyrstart = pars.PYR0 * abs( PYRtune )
        gypstart = pars.GYP0 * abs( GYPtune )

    #else use pre-tuned (Gtune.key=0)
    if Gtune.key == 0:
        outputs = [0.55, 1, 1.2, 1, 0.1, 0.05, 3]
        gstart = pars.G0 * outputs[0]
        cstart = pars.C0 * outputs[1]
        pyrstart = pars.PYR0 * outputs[2]
        gypstart = pars.GYP0 * outputs[3]
        ostart = pars.O0 * outputs[4]
        sstart = pars.S0 * outputs[5]
        astart = pars.A0 * outputs[6]

    #### define starting parameters here ###
    start_pars = SCION_classes.Starting_parameters_class(
                                           pstart, tempstart, CAL_start, N_start,
                                           OSr_start, SSr_start, delta_A_start,
                                           delta_S_start, delta_G_start, delta_C_start,
                                           delta_PYR_start, delta_GYP_start,
                                           delta_OSr_start, delta_SSr_start,
                                           ostart, astart, sstart, gstart, cstart,
                                           pyrstart,gypstart)
    ### note model start time
    model_time = time.time()

    #make classes for results storage
    if sensanal.key == 0:
        workingstate = SCION_classes.Workingstate_class()
        gridstate_array = np.zeros([40,48,22], dtype=complex)
        gridstate = SCION_classes.Gridstate_class(gridstate_array)

        ##### run the system
        rawoutput = solve_ivp(SCION_equations.SCION_equations, [model_pars.whenstart,
                                                                model_pars.whenend],
                                  start_pars.startstate, method='BDF',
                                  max_step=1e6, args=[pars, forcings, sensanal,
                                                      INTERPSTACK, model_pars, workingstate,
                                                      stepnumber, gridstate, sensparams,
                                                      erosion_pars])
    else:
        workingstate = SCION_classes.Workingstate_class_sensanal()
        #we don't need gridstates for sensanal run, so pass an empty array
        gridstate = np.asarray([])
        ##### run the system
        rawoutput = solve_ivp(SCION_equations.SCION_equations, [model_pars.whenstart,
                                                                model_pars.whenend],
                                  start_pars.startstate, method='BDF',
                                  max_step=1e6, args=[pars, forcings, sensanal,
                                                      INTERPSTACK, model_pars, workingstate,
                                                      stepnumber, gridstate, sensparams,
                                                      erosion_pars])

    #####################################################################
    #################   Postprocessing   ################################
    #####################################################################
    #print(sensparams.randminusplus1)
    #### size of output
    model_pars.output_length = len(rawoutput.t)

    if sensanal.key == 0:
        #### model finished output to screen
        print('Integration finished \t')
        print('Total steps: %d \t' % stepnumber.step)
        print('Output steps: %d \n' % model_pars.output_length)

    #### print final model states using final state for each timepoint
    #### during integration

    if sensanal.key == 0:
        print('assembling   vectors... \t')

    #### trecords is index of shared values between ode15s output T vector and
    #### model recorded workingstate t vector
    common_vals, workingstate_index, rawouput_index = intersect_mtlb(workingstate.time,rawoutput.t)

    #get field names to make our final result class
    field_names = []
    for property, value in vars(workingstate).items():
        field_names.append(property)

    #convert our workingstates to arrays for indexing and get our state class
    if sensanal.key == 0:
        workingstate.convert_to_array()
        state = SCION_classes.State_class(workingstate, workingstate_index)
        run = SCION_classes.Run_class(state, gridstate, pars, model_pars,
                                      start_pars, forcings, erosion_pars)

    else:
        workingstate.convert_to_array()
        state = SCION_classes.State_class_sensanal(workingstate, workingstate_index)
        run = SCION_classes.Run_class(state, gridstate, pars, model_pars,
                                      start_pars, forcings, erosion_pars)

    if sensanal.key == 0:
        #### done message
        print('Done')
        endtime = time.time() - start_time
        print('time:', endtime)

    #####################################################################
    ###########################   Plotting   ###########################
    ####################################################################
    ### only plot if no tuning structure exists, only plot fluxes for quick runs
    #also needs fixing

    if Gtune.key == 0: # == 1
        if plotrun.key == 1: # == 1
            SCION_plot_fluxes.SCION_plot_fluxes(state, model_pars, pars)
            if S>-1:
                SCION_plot_worldgraphic.SCION_plot_worldgraphic(gridstate, INTERPSTACK)

    if sensanal.key == 1:
        return run
    else:
        return run, INTERPSTACK
