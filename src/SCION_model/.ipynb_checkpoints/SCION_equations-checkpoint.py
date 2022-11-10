import numpy as np
import metpy
import metpy.calc as mpcalc
from scipy.stats import norm
from scipy.interpolate import interp1d

def SCION_equations(t,y, pars, forcings, sensanal, INTERPSTACK,
                         model_pars, workingstate, stepnumber, gridstate,
                         sensparams, erosion_pars):

    #################################################################################################
    #                                                                                               #
    #              110111010                                                                        #
    #           111010-1-----101                                                                    #
    #        1011111---------101111                                                                 #
    #      11011------------------101         pySCION: Spatial Continuous Integration               #
    #     111-----------------10011011        Earth Evolution Model (for python!)                   #
    #    1--10---------------1111011111                                                             #
    #    1---1011011---------1010110111       Coded by Benjaw J. W. Mills                         #
    #    1---1011000111----------010011       email: b.mills@leeds.ac.uk                            #
    #    1----1111011101----------10101       Translated into python by Andrew Merdith              #
    #     1----1001111------------0111        Model equations file                                  #
    #      1----1101-------------1101         contains flux and reservoir equations                 #
    #        1--111----------------1                                                                #
    #           1---------------1                                                                   #
    #               111011011                                                                       #
    #################################################################################################

    #numpy divides for all elements in an array, even the ones that aren't selected by the where (which are 0 and throwing the errow).
    np.seterr(divide = 'ignore')
    ###### setup dy array
    dy = np.zeros(21)

    ############ get variables from Y to make working easier
    P = y[0]
    O = y[1]
    A = y[2]
    S = y[3]
    G = y[4]
    C = y[5]
    PYR = y[6]
    GYP = y[7]
    # TEMP = y[8]
    # CAL = y[9]
    N = y[10]
    OSr = y[17]
    SSr = y[19]
    dSSr = y[20]/y[19]

    #### geological time in Ma
    t_geol = t*(1e-6)

    ####### calculate isotopic fractionation of reservoirs
    delta_G = y[11]/y[4]
    delta_C = y[12]/y[5]
    delta_GYP  = y[14]/y[7]
    delta_PYR  = y[13]/y[6]

    ####### atmospheric fraction of total CO2, atfrac(A)
    atfrac0 = 0.01614
    ####### constant
    # atfrac = 0.01614
    ####### variable
    atfrac = atfrac0 * (A/pars.A0)

    ######## calculations for pCO2, pO2
    RCO2 = (A/pars.A0)*(atfrac/atfrac0)
    CO2atm = RCO2*(280e-6)
    CO2ppm = RCO2*280

    ##### mixing ratio of oxygen (not proportional to O reservoir)
    mrO2 = ( O/pars.O0 )  /   ( (O/pars.O0)  + pars.copsek16 )
    ##### relative moles of oxygen
    RO2 =  O/pars.O0

    #####################################################################
    ############   Interpolate forcings for this timestep   #############
    #####################################################################

    #### COPSE Reloaded forcing set
    #interp1d in the format: E_reloaded = interp1d(1e6 * forcings.t, forcings.E)(t)
    #is equivalent to
    #E_interp1d = interp1d(1e6 * forcings.t, forcings.E)
    #E_reloaded = E_interp1d(t)
    ##NB the interpretator could be generated in initialise and then accessed later with the
    #correct timestep, using the above format
    E_reloaded = forcings.E_reloaded_INTERP(t)#interp1d(1e6 * forcings.t, forcings.E)(t)
    W_reloaded = forcings.W_reloaded_INTERP(t)#interp1d(1e6 * forcings.t, forcings.W)(t)
    #### Additional forcings
    GR_BA = forcings.GR_BA_reloaded_INTERP(t)
    newGA = forcings.newGA_reloaded_INTERP(t)

    #
    #select degassing here
    #

    if pars.DEGASS_test == 0:
        D_combined_mid = forcings.D_complete_SMOOTH_INTERP(t_geol)#1
        D_combined_min = forcings.D_complete_min_SMOOTH_INTERP(t_geol)#0.9
        D_combined_max = forcings.D_complete_max_SMOOTH_INTERP(t_geol)#1.1

    else:
        D_combined_mid = 1
        D_combined_min = 1
        D_combined_max = 1
    ######################################################################
    ####################  Choose forcing functions  ######################
    ######################################################################

    ######################################################################
    #DEGASS = 1 ;
    DEGASS = D_combined_mid
    ######################################################################
    W = W_reloaded
    ######################################################################
    EVO = E_reloaded
    ######################################################################
    CPLAND = 1
    # CPLAND = CP_reloaded ;
    ######################################################################
    #Bforcing = interp1d([-1000, -150, 0],[0.75, 0.75, 1])(t_geol)
    #Bforcing = interp1d([-1000, -120,-100,-50, 0],[0.82, 0.82, 0.85, 0.90, 1])(t_geol)
    Bforcing = 1
    carb_forcing = 1.5
    ######################################################################
    BAS_AREA = GR_BA
    ######################################################################
    GRAN_AREA = newGA
    ######################################################################
    PREPLANT = 1/4 #contribution to CO2 prior to plants
    capdelS = 27
    capdelC_land = 27
    capdelC_marine = 35
    ######################################################################

    #### SHORELINE
    SHORELINE = forcings.shoreline_INTERP(t_geol)

    #### bioturbation forcing

    f_biot = forcings.f_biot_INTERP(t)
    CB = forcings.CB_INTERP(f_biot)
    
    #biome forcing
    biome_forcing = forcings.biome_INTERP(t)
    
    ######################################################################
    ######################   Sensitivity analysis  ######################
    ######################################################################

    #### all sensparams vary between [-1 +1]
    if sensanal.key == 1:
        #print('sensanal, here')
        ##### Very degassing between upper and lower bounds
        if sensparams.randminusplus1 > 0:
            DEGASS = (1 - sensparams.randminusplus1)*DEGASS + sensparams.randminusplus1*D_combined_max
        else:
            DEGASS = (1 + sensparams.randminusplus1)*DEGASS - sensparams.randminusplus1*D_combined_min

        #### simple +/- 20% variation
        BAS_AREA = BAS_AREA * (1 + 0.2*sensparams.randminusplus2)
        GRAN_AREA = GRAN_AREA * (1 + 0.2*sensparams.randminusplus3)

        #### preplant varies from 1/1 to 1/7
        PREPLANT = 1 / ( 4 + 3*sensparams.randminusplus4)#1/4

        ####
        capdelS = 30 + 10*sensparams.randminusplus5
        capdelC_land = 25 + 5*sensparams.randminusplus6
        capdelC_marine = 30 + 5*sensparams.randminusplus7

    ######################################################################
    ##################### Spatial fields from stack   ####################
    ######################################################################

    ######################################################################
    ######################   Fetch keyframe grids   ######################
    ######################################################################

    #### find past and future keyframes
    key_future_time = np.min(INTERPSTACK.time[(INTERPSTACK.time - t_geol) >=0])
    #because 600 Ma is older than our times, it can return an empty array which breaks np.max, so we check
    temp_key_past_time = INTERPSTACK.time[INTERPSTACK.time-t_geol <=0]
    if not temp_key_past_time.size > 0:
        #if empty, i.e. between 600 and 540 Ma
        key_past_time = key_future_time
    else:
        key_past_time = np.max(temp_key_past_time)

    #### find keyframe indexes and fractional contribution
    key_past_index = np.argwhere(INTERPSTACK.time == key_past_time )[0][0]
    key_future_index = np.argwhere(INTERPSTACK.time == key_future_time )[0][0]
    dist_to_past = np.abs(key_past_time - t_geol)
    dist_to_future = np.abs(key_future_time - t_geol)

    #### fractional contribution of each keyframe
    #only if time = 0?
    if dist_to_past + dist_to_future == 0:
        contribution_past = 1
        contribution_future = 0
    else:
        contribution_past = dist_to_future / ( dist_to_past + dist_to_future )
        contribution_future = dist_to_past / ( dist_to_past + dist_to_future )

    #### intrepolate keyframe CO2 concentrations to generate keyframe fields
    #### find INTERPSTACK keyframes using model CO2
    #print(CO2ppm)
    if CO2ppm > 112000:
        print('too bloody hot clive', CO2ppm)
        CO2ppm = 111999

    key_upper_CO2 = np.min(INTERPSTACK.CO2[(INTERPSTACK.CO2 - CO2ppm) >= 0])
    #if CO2 is between 0 and 10 it throws a value error. So this makes it grab the smallest
    #co2 value
    try:
        key_lower_CO2 = np.max(INTERPSTACK.CO2[(INTERPSTACK.CO2 - CO2ppm) <= 0])
    except ValueError:
        key_lower_CO2 = 10
    #### find keyframe indexes and fractional contribution
    key_upper_CO2_index = np.argwhere(INTERPSTACK.CO2 == key_upper_CO2 )[0][0]
    key_lower_CO2_index = np.argwhere(INTERPSTACK.CO2 == key_lower_CO2 )[0][0]
    dist_to_upper = np.abs(key_upper_CO2 - CO2ppm)
    dist_to_lower = np.abs(key_lower_CO2 - CO2ppm)

    #### fractional contribution of each keyframe
    if dist_to_upper + dist_to_lower == 0:
        contribution_lower = 1
        contribution_upper = 0
    else:
        contribution_upper = dist_to_lower / (dist_to_upper + dist_to_lower)
        contribution_lower = dist_to_upper / (dist_to_upper + dist_to_lower)

    ######################################################################
    ###### Create time keyframes using CO2 keyfield contributions   ######
    ######################################################################

    #### Runoff
    RUNOFF_past = (contribution_upper * np.copy(INTERPSTACK.runoff[:, :, key_upper_CO2_index, key_past_index]) + \
                  contribution_lower * np.copy(INTERPSTACK.runoff[:, :, key_lower_CO2_index, key_past_index]))
    RUNOFF_future = (contribution_upper * np.copy(INTERPSTACK.runoff[:, :, key_upper_CO2_index, key_future_index]) + \
                    contribution_lower * np.copy(INTERPSTACK.runoff[:, :, key_lower_CO2_index, key_future_index]))

    #### Tair
    Tair_past = (contribution_upper * np.copy(INTERPSTACK.Tair[:, :, key_upper_CO2_index, key_past_index]) + \
                     contribution_lower * np.copy(INTERPSTACK.Tair[:, :, key_lower_CO2_index, key_past_index]))
    Tair_future = (contribution_upper * np.copy(INTERPSTACK.Tair[:, :, key_upper_CO2_index, key_future_index]) + \
                   contribution_lower * np.copy(INTERPSTACK.Tair[:, :, key_lower_CO2_index, key_future_index]))

    #### time kayframes that don't depend on CO2
    #### Topography
    TOPO_past = np.copy(INTERPSTACK.topo[:,:,key_past_index])
    TOPO_future = np.copy(INTERPSTACK.topo[:,:,key_future_index])

    #### slope
    tslope_past = np.copy(INTERPSTACK.slope[:,:,key_past_index])
    tslope_future = np.copy(INTERPSTACK.slope[:,:,key_future_index])

    ###ARC_past
    ARC_past = np.copy(INTERPSTACK.arc[:,:,key_past_index])
    ARC_future = np.copy(INTERPSTACK.arc[:,:,key_future_index])

    arc_mask_past = np.copy(INTERPSTACK.arc_mask[:,:,key_past_index])
    arc_mask_future = np.copy(INTERPSTACK.arc_mask[:,:,key_future_index])

    arc_enhancement_past = np.copy(INTERPSTACK.arc_enhancement[:,:,key_past_index])
    arc_enhancement_future = np.copy(INTERPSTACK.arc_enhancement[:,:,key_future_index])

    #### Sutures
    SUTURE_past = np.copy(INTERPSTACK.suture[:,:,key_past_index])
    SUTURE_future = np.copy(INTERPSTACK.suture[:,:,key_future_index])

    suture_mask_past = np.copy(INTERPSTACK.suture_mask[:,:,key_past_index])
    suture_mask_future = np.copy(INTERPSTACK.suture_mask[:,:,key_future_index])

    suture_enhancement_past = np.copy(INTERPSTACK.suture_enhancement[:,:,key_past_index])
    suture_enhancement_future = np.copy(INTERPSTACK.suture_enhancement[:,:,key_future_index])

    #### relict arcs
    RELICT_past = np.copy(INTERPSTACK.relict_arc[:,:,key_past_index])
    RELICT_future = np.copy(INTERPSTACK.relict_arc[:,:,key_future_index])

    relict_arc_mask_past = np.copy(INTERPSTACK.relict_arc_mask[:,:,key_past_index])
    relict_arc_mask_future = np.copy(INTERPSTACK.relict_arc_mask[:,:,key_future_index])

    relict_arc_enhancement_past = np.copy(INTERPSTACK.relict_arc_enhancement[:,:,key_past_index])
    relict_arc_enhancement_future = np.copy(INTERPSTACK.relict_arc_enhancement[:,:,key_future_index])
    
    #root depths
    root_depths_past = np.copy(INTERPSTACK.root_depths[:,:,key_past_index])
    root_depths_future = np.copy(INTERPSTACK.root_depths[:,:,key_future_index])

    root_depth_enhancement_past = np.copy(INTERPSTACK.root_enhancement[:,:,key_past_index])
    root_depth_enhancement_future = np.copy(INTERPSTACK.root_enhancement[:,:,key_future_index])
    
    
    #### last keyframe land recorded for plot
    land_past = np.copy(INTERPSTACK.land[:,:,key_past_index])
    land_future = np.copy(INTERPSTACK.land[:,:,key_future_index])

    #### gridbox area
    GRID_AREA_km2 = np.copy(INTERPSTACK.gridarea)

    ######################################################################
    ##################   Spatial silicate weathering   ###################
    ######################################################################

    #### West / Maffre weathering approximation

    #### runoff in mm/yr
    #have to np.copy otherwise original gets altered
    Q_past = np.copy(RUNOFF_past)
    Q_past[Q_past<0] = 0
    Q_future = np.copy(RUNOFF_future)
    Q_future[Q_future<0] = 0

    #### temp in kelvin
    T_past = Tair_past + 273
    T_future = Tair_future + 273

    #### pierre erosion calculation, t/m2/yr
    k_erosion = 3.3e-3 #### for 16Gt present day erosion in FOAM
    EPSILON_past = k_erosion * (Q_past**0.31) * tslope_past * np.maximum(Tair_past,2)
    EPSILON_future = k_erosion * (Q_future**0.31) * tslope_future * np.maximum(Tair_future,2)
    #### check total tonnes of erosion - should be ~16Gt
    EPSILON_per_gridbox_past = EPSILON_past * GRID_AREA_km2 * 1e6 ### t/m2/yr * m2
    EPSILON_per_gridbox_future = EPSILON_future * GRID_AREA_km2 * 1e6 ### t/m2/yr * m2
    erosion_tot_past = np.sum(EPSILON_per_gridbox_past)
    erosion_tot_future = np.sum(EPSILON_per_gridbox_future)
    erosion_tot = erosion_tot_past*contribution_past + erosion_tot_future*contribution_future

    #### Pierre weathering equation params
    Xm = erosion_pars.Xm
    K = erosion_pars.K
    kw = erosion_pars.kw
    Ea = erosion_pars.Ea
    z = erosion_pars.z
    sigplus1 = erosion_pars.sigplus1
    T0 = erosion_pars.T0
    R = erosion_pars.R

    #### equations
    R_T_past = np.exp((Ea / (R * T0)) - (Ea / (R * T_past)))
    R_T_future = np.exp((Ea / ( R * T0)) - (Ea / (R * T_future)))
    R_Q_past = 1 - np.exp(-1 * kw * Q_past)
    R_Q_future = 1 - np.exp(-1 * kw * Q_future)
    R_reg_past = ((z / EPSILON_past)**sigplus1 ) / sigplus1
    R_reg_future = ((z / EPSILON_future)**sigplus1 ) / sigplus1

    #### equation for CW per km2 in each box
    #consider this the 'base' weathering
    CW_per_km2_past_raw = 1e6 * EPSILON_past * Xm * (1 - np.exp(-1 * K * R_Q_past * R_T_past * R_reg_past))
    CW_per_km2_future_raw = 1e6 * EPSILON_future * Xm * (1 - np.exp(-1 * K * R_Q_future * R_T_future * R_reg_future))

    #we need to find the non arc and suture weathering and subtract it from
    #our final calculation so we don't double dip
    CW_per_km2_past_raw_AF = CW_per_km2_past_raw * arc_enhancement_past
    CW_per_km2_future_raw_AF = CW_per_km2_future_raw * arc_enhancement_future

    #this is weathering attributed to sutures *only*
    CW_per_km2_past_raw_SF = CW_per_km2_past_raw * suture_enhancement_past
    CW_per_km2_future_raw_SF = CW_per_km2_future_raw * suture_enhancement_future

    #this is weathering attributed to relict arcs *only*
    CW_per_km2_past_raw_RAF = CW_per_km2_past_raw * relict_arc_enhancement_past
    CW_per_km2_future_raw_RAF = CW_per_km2_future_raw * relict_arc_enhancement_future

    #this is weathering attributed to roots *only*
    CW_per_km2_past_raw_ROOTS = CW_per_km2_past_raw * root_depth_enhancement_past
    CW_per_km2_future_raw_ROOTS = CW_per_km2_future_raw * root_depth_enhancement_future


    #mutliply our 'base' weathering layer by masks to get only weathering
    #where there's no arcs and Sutures
    non_arc_suture_weathering_past = CW_per_km2_past_raw * (arc_mask_past != True) * (suture_mask_past != True) * (relict_arc_mask_past != True)
    non_arc_suture_weathering_future = CW_per_km2_future_raw * (arc_mask_future != True) * (suture_mask_future != True) * (relict_arc_mask_future != True)

    #now add them together
    CW_per_km2_past = CW_per_km2_past_raw_SF + CW_per_km2_past_raw_AF + CW_per_km2_past_raw_RAF + non_arc_suture_weathering_past
    CW_per_km2_future = CW_per_km2_future_raw_SF + CW_per_km2_future_raw_AF + CW_per_km2_future_raw_RAF + non_arc_suture_weathering_future

    #get present day weathering to calculate silw_scale
    CO2ppm_present_day = 280

    if stepnumber.step == 1:

        pars.get_CW_present(CO2ppm_present_day, INTERPSTACK, k_erosion, Xm, K,
        kw, Ea, z, sigplus1, T0, R,GRID_AREA_km2)

    #### CW total
    CW_past = CW_per_km2_past * GRID_AREA_km2
    CW_future = CW_per_km2_future * GRID_AREA_km2
    ### world CW
    CW_past[np.isnan(CW_past)==1] = 0
    CW_future[np.isnan(CW_future)==1] = 0
    CW_sum_past = np.sum(CW_past)
    CW_sum_future = np.sum(CW_future)
    CW_tot = CW_sum_past * contribution_past + CW_sum_future * contribution_future

    #### carbonate weathering spatial approximation, linear with runoff
    if pars.PGEOG_test == 1:
        k_carb_scale = 200
    else:
        k_carb_scale = 200 #### scaling parameter to recover present day rate
    CWcarb_per_km2_past = k_carb_scale * Q_past
    CWcarb_per_km2_future = k_carb_scale * Q_future
    #### CW total
    CWcarb_past = CWcarb_per_km2_past * GRID_AREA_km2
    CWcarb_future = CWcarb_per_km2_future * GRID_AREA_km2
    #### world CWcarb
    CWcarb_past[np.isnan(CWcarb_past)==1] = 0
    CWcarb_future[np.isnan(CWcarb_future)==1] = 0
    CWcarb_sum_past = np.sum(CWcarb_past)
    CWcarb_sum_future = np.sum(CWcarb_future)

    ######################################################################
    ##################   Grid interpolated variables   ###################
    ######################################################################

    #### silicate weathering scale factor by present day rate in cation tonnes
    silw_scale = pars.CW_present #### default run 4.2e8 at suturefactor = 1; preplant =1/4; for k erosion 3.3e-3. #8.5e8 for palaeogeog=1
    #### overall spatial weathering
    silw_spatial = CW_tot * ((pars.k_basw + pars.k_granw) / silw_scale)
    carbw_spatial = (CWcarb_sum_past*contribution_past + CWcarb_sum_future*contribution_future)

    ### global average surface temperature
    GAST = np.mean(Tair_past * model_pars.rel_contrib)*contribution_past  +  np.mean(Tair_future * model_pars.rel_contrib)*contribution_future

    #### set assumed ice temperature
    Tcrit = -10
    #### ice line calculations
    Tair_past_ice = np.copy(Tair_past)
    Tair_past_ice[Tair_past_ice >= Tcrit] = 0
    Tair_past_ice[Tair_past_ice < Tcrit] = 1
    Tair_future_ice = np.copy(Tair_future)
    Tair_future_ice[Tair_future_ice >= Tcrit] = 0
    Tair_future_ice[Tair_future_ice < Tcrit] = 1
    #### count only continental ice
    Tair_past_ice = Tair_past_ice * land_past
    Tair_future_ice = Tair_future_ice * land_future
    ### sum into lat bands
    latbands_past = np.sum(Tair_past_ice, axis=1)
    latbands_future = np.sum(Tair_future_ice, axis=1)
    latbands_past[latbands_past>0] = 1
    latbands_future[latbands_future>0] = 1
    ### find appropiate lat
    latresults_past =  INTERPSTACK.lat * latbands_past
    latresults_future =  INTERPSTACK.lat * latbands_future
    latresults_past[latresults_past == 0] = 90
    latresults_future[latresults_future == 0] = 90
    #### lowest glacial latitude
    iceline_past = np.min(np.abs(latresults_past))
    iceline_future = np.min(np.abs(latresults_future))
    iceline = iceline_past * contribution_past +  iceline_future * contribution_future

    ######################################################################
    ########################   Global variables   ########################
    ######################################################################

    #### effect of temp on VEG %%%% fixed
    V_T = 1 - (((GAST - 15) / 25) **2)

    ### effect of CO2 on VEG
    P_atm = CO2atm * 1e6
    P_half = 183.6
    P_min = 10
    V_co2 = (P_atm - P_min) / (P_half + P_atm - P_min)

    #### effect of O2 on VEG
    V_o2 = 1.5 - 0.5*(O/pars.O0)

    #### full VEG limitation
    V_npp = 2*EVO*V_T*V_o2*V_co2

    #### COPSE reloaded fire feedback
    ignit = min(max(48*mrO2 - 9.08 , 0), 5 )
    firef = pars.kfire/(pars.kfire - 1 + ignit)

    #### Mass of terrestrial biosphere
    VEG = V_npp * firef

    #### basalt and granite temp dependency - direct and runoff
    Tsurf = GAST + 273
    TEMP_gast = Tsurf

    #root weathering
    rootw = INTERPSTACK.root_depth/460
    #### COPSE reloaded fbiota
    V = VEG
    f_biota = ((1 - min(V*W, 1)) * PREPLANT * (RCO2**0.5) + (V*W)) * rootw

    #### version using gran area and conserving total silw
    #basw+ granw should equal total weathering rate, irrispective of basw/granw fractions

    basw = silw_spatial * (pars.basfrac * BAS_AREA / (pars.basfrac * BAS_AREA + (1 - pars.basfrac) * GRAN_AREA))
    granw = silw_spatial * ((1 - pars.basfrac) * GRAN_AREA / (pars.basfrac * BAS_AREA + (1 - pars.basfrac) * GRAN_AREA))



    #### add fbiota
    basw = basw * f_biota
    granw = granw * f_biota
    carbw = carbw_spatial * f_biota

    #### overall weathering
    silw = basw + granw
    carbw_relative = (carbw/pars.k_carbw)

    #### oxidative weathering
    oxidw = pars.k_oxidw*carbw_relative*(G/pars.G0)*((O/pars.O0)**pars.a)

    #### pyrite weathering
    pyrw = pars.k_pyrw*carbw_relative*(PYR/pars.PYR0)

    #### gypsum weathering
    gypw = pars.k_gypw*(GYP/pars.GYP0)*carbw_relative

    #### seafloor weathering, revised following Brady and Gislason but not directly linking to CO2
    f_T_sfw = np.exp(0.0608*(Tsurf-288))
    sfw = pars.k_sfw * f_T_sfw * DEGASS ### assume spreading rate follows degassing here
    ######## Degassing
    ocdeg = pars.k_ocdeg*DEGASS*(G/pars.G0)
    ccdeg = pars.k_ccdeg*DEGASS*(C/pars.C0)*Bforcing
    pyrdeg = pars.k_pyrdeg*(PYR/pars.PYR0)*DEGASS
    gypdeg = pars.k_gypdeg*(GYP/pars.GYP0)*DEGASS

    #### gypsum burial
    #mgsb = pars.k_mgsb*(S/pars.S0)
    mgsb = pars.k_mgsb*(S/pars.S0)*(1/SHORELINE)

    #### carbonate burial
    mccb = carbw + silw

    #### COPSE reloaded Phospherous weathering
    pfrac_silw = 0.8
    pfrac_carbw = 0.14
    pfrac_oxidw = 0.06

    ###Extra phospherous weathering from Jack Longman 2021 Nat Geo
    #95th percentile + 95th percentile 2 Myr weathering, x 5 for recycling
    P_GICE = 5*( 6.49e15 + 2*1.23e15 )
    P_HICE = 5*( 8.24e15 + 2*1.23e15 )
    EXTRA_P = P_GICE*1e-6*norm.pdf(t_geol,-453.45,0.4)/norm.pdf(-453.45,-453.45,0.4) + P_HICE*1e-6*norm.pdf(t_geol,-444,0.4)/norm.pdf(-444,-444,0.4)
    phosw = EXTRA_P + pars.k_phosw*((pfrac_silw)*(silw/pars.k_silw) + (pfrac_carbw)*(carbw/pars.k_carbw) + (pfrac_oxidw)*(oxidw/pars.k_oxidw))

    #### COPSE reloaded
    pland = pars.k_landfrac * VEG * phosw
    pland0 = pars.k_landfrac * pars.k_phosw
    psea = phosw - pland

    #### convert total reservoir moles to micromoles/kg concentration
    Pconc = (P/pars.P0) * 2.2
    Nconc = (N/pars.N0) * 30.9
    newp = 117 * min(Nconc/16,Pconc)

    #### carbon burial
    mocb = pars.k_mocb*((newp/pars.newp0)**pars.b) * CB
    locb = pars.k_locb*(pland/pland0)*CPLAND

    #### PYR burial function (COPSE)
    fox= 1/(O/pars.O0)
    #### mpsb scales with mocb so no extra uplift dependence
    mpsb = pars.k_mpsb*(S/pars.S0)*fox*(mocb/pars.k_mocb)

    #### OCEAN ANOXIC FRACTION
    k_anox = 12
    k_u = 0.5
    ANOX = 1 / (1 + np.exp(-1 * k_anox * (k_u * (newp/pars.newp0) - (O/pars.O0))))

    #### nutrient burial
    CNsea = 37.5
    monb = mocb/CNsea

    #### P burial with bioturbation on
    CPbiot = 250
    CPlam = 1000
    mopb = mocb*((f_biot/CPbiot) + ((1-f_biot)/CPlam))
    capb = pars.k_capb*(mocb/pars.k_mocb)

    #### reloaded
    fepb = (pars.k_fepb/pars.k_oxfrac)*(1-ANOX)*(P/pars.P0)

    #### nitrogen cycle
    #### COPSE reloaded
    if (N/16) < P:
        nfix = pars.k_nfix * (((P - (N/16)) / (pars.P0 - (pars.N0/16)))**2)
    else:
        nfix = 0

    denit = pars.k_denit * (1 + (ANOX / (1-pars.k_oxfrac))) * (N/pars.N0)

    #### reductant input
    reductant_input = pars.k_reductant_input * DEGASS

    ######################################################################
    ######################   Reservoir calculations  #####################
    ######################################################################

    ### Phosphate
    dy[0] = psea - mopb - capb - fepb

    ### Oxygen
    dy[1] = locb + mocb - oxidw  - ocdeg  + 2*(mpsb - pyrw  - pyrdeg) - reductant_input

    ### Carbon dioxide
    dy[2] = -locb - mocb + oxidw + ocdeg + ccdeg + carbw - mccb - sfw  + reductant_input

    ### Sulphate
    dy[3] = gypw + pyrw - mgsb - mpsb + gypdeg + pyrdeg

    ### Buried organic C
    dy[4] = locb + mocb - oxidw - ocdeg

    ### Buried carb C
    dy[5] = mccb + sfw - carbw - ccdeg

    ### Buried pyrite S
    dy[6] = mpsb - pyrw - pyrdeg

    ### Buried gypsum S
    dy[7] = mgsb - gypw - gypdeg

    ### Nitrate
    dy[10] = nfix - denit - monb


    ######################################################################
    #######################   Isotope reservoirs  ########################
    ######################################################################

    ### d13c and d34s for forwards model
    d13c_A = y[15] / y[2]
    d34s_S = y[16] / y[3]

    ### carbonate fractionation
    delta_locb = d13c_A - capdelC_land
    delta_mocb = d13c_A - capdelC_marine
    delta_mccb = d13c_A
    ### S isotopes (copse)
    delta_mpsb = d34s_S - capdelS

    ### deltaORG_C*ORG_C
    dy[11] =  locb*(delta_locb) + mocb*(delta_mocb) - oxidw*delta_G - ocdeg*delta_G

    ### deltaCARB_C*CARB_C
    dy[12] =  mccb*delta_mccb + sfw*delta_mccb - carbw*delta_C - ccdeg*delta_C

    ### deltaPYR_S*PYR_S (young)
    dy[13] =  mpsb*(delta_mpsb) - pyrw*delta_PYR - pyrdeg*delta_PYR

    ### deltaGYP_S*GYP_S (young)
    dy[14] =  mgsb*d34s_S - gypw*delta_GYP - gypdeg*delta_GYP

    ### delta_A * A
    dy[15] = -locb*(delta_locb) -mocb*(delta_mocb) + oxidw*delta_G + ocdeg*delta_G + ccdeg*delta_C + carbw*delta_C - mccb*delta_mccb - sfw*delta_mccb + reductant_input*-5

    ### delta_S * S
    dy[16] = gypw*delta_GYP + pyrw*delta_PYR -mgsb*d34s_S - mpsb*(delta_mpsb) + gypdeg*delta_GYP + pyrdeg*delta_PYR

    ######################################################################
    ########################   Strontium system   ########################
    ######################################################################

    ### fluxes
    Sr_granw = pars.k_Sr_granw *(granw / pars.k_granw)
    Sr_basw = pars.k_Sr_basw *(basw / pars.k_basw)
    Sr_sedw = pars.k_Sr_sedw *(carbw / pars.k_carbw) * (SSr/pars.SSr0)
    Sr_mantle = pars.k_Sr_mantle * DEGASS
    Sr_sfw = pars.k_Sr_sfw * (sfw/pars.k_sfw) * (OSr/pars.OSr0)
    Sr_metam = pars.k_Sr_metam * DEGASS * (SSr/pars.SSr0)
    Sr_sedb = pars.k_Sr_sedb * (mccb/pars.k_mccb) * (OSr/pars.OSr0)

    #### fractionation calculations
    delta_OSr = y[18] / y[17] ;
    delta_SSr = y[20] / y[19] ;

    #### original frac
    RbSr_bas = 0.1
    RbSr_gran = 0.26
    RbSr_mantle = 0.066
    RbSr_carbonate = 0.5

    #### frac calcs
    dSr0 = 0.69898
    tforwards = 4.5e9 + t
    lambda_val = 1.4e-11
    dSr_bas = dSr0 + RbSr_bas*(1 - np.exp(-1*lambda_val*tforwards))
    dSr_gran = dSr0 + RbSr_gran*(1 - np.exp(-1*lambda_val*tforwards))
    dSr_mantle = dSr0 + RbSr_mantle*(1 - np.exp(-1*lambda_val*tforwards))

    #### Ocean [Sr]
    dy[17] = Sr_granw + Sr_basw + Sr_sedw + Sr_mantle - Sr_sedb - Sr_sfw

    #### Ocean [Sr]*87/86Sr
    dy[18] = Sr_granw*dSr_gran + Sr_basw*dSr_bas + Sr_sedw*delta_SSr + Sr_mantle*dSr_mantle - Sr_sedb*delta_OSr - Sr_sfw*delta_OSr

    #### Sediment [Sr]
    dy[19] = Sr_sedb - Sr_sedw - Sr_metam

    #### Sediment [Sr]*87/86Sr
    dy[20] = Sr_sedb*delta_OSr - Sr_sedw*delta_SSr - Sr_metam*delta_SSr + SSr*lambda_val*RbSr_carbonate*np.exp(lambda_val*tforwards)


    ######################################################################
    #####################   Mass conservation check   ####################
    ######################################################################

    res_C = A + G + C
    res_S = S + PYR + GYP
    iso_res_C = A*d13c_A + G*delta_G + C*delta_C
    iso_res_S = S*d34s_S + PYR*delta_PYR + GYP*delta_GYP

    ######################################################################
    ################   Record states for single run   ################
    ######################################################################

    if sensanal.key == 0:
        new_data = [iso_res_C, iso_res_S, res_C, res_S, t, TEMP_gast, TEMP_gast - 273, P, O, A, S, G, C, PYR, GYP,N, OSr, SSr, d13c_A, delta_mccb, d34s_S, delta_G, delta_C, delta_PYR, delta_GYP, delta_OSr,
        DEGASS, W,EVO, CPLAND, Bforcing, BAS_AREA, GRAN_AREA, RCO2, RO2, mrO2, VEG, ANOX, iceline,
        mocb, locb, mccb, mpsb, mgsb, silw, carbw, oxidw, basw, granw, phosw, psea, nfix, denit, VEG,
        pyrw, gypw, ocdeg, ccdeg, pyrdeg, gypdeg, sfw, Sr_granw, Sr_basw, Sr_sedw, Sr_mantle, dSSr,
        newp/pars.newp0, erosion_tot, t_geol]

        workingstate.add_workingstates(new_data)
        ### print a gridstate when each keytime threshold is crossed, or at model end
        next_stamp = model_pars.next_gridstamp
        if model_pars.finishgrid == 0:
            if t_geol > next_stamp or t_geol == 0:
                #### write gridstates
                gridstate.time_myr[:,:,model_pars.gridstamp_number] = np.copy(next_stamp)
                gridstate.land[:,:,model_pars.gridstamp_number] = np.copy(land_past)
                gridstate.SUTURE[:,:,model_pars.gridstamp_number] = np.copy(SUTURE_past)
                gridstate.ARC[:,:,model_pars.gridstamp_number] = np.copy(ARC_past)
                gridstate.RELICT_ARC[:,:,model_pars.gridstamp_number] = np.copy(RELICT_past)
                gridstate.Q[:,:,model_pars.gridstamp_number] = np.copy(Q_past)
                gridstate.Tair[:,:,model_pars.gridstamp_number] = np.copy(Tair_past)
                gridstate.TOPO[:,:,model_pars.gridstamp_number] = np.copy(TOPO_past)
                gridstate.CW[:,:,model_pars.gridstamp_number] = np.copy(CW_per_km2_past) #t/km2/yr
                gridstate.CWcarb[:,:,model_pars.gridstamp_number] = np.copy(CWcarb_past)
                gridstate.EPSILON[:,:,model_pars.gridstamp_number] = np.copy(EPSILON_past) * 1e6 #t/km2/yr

                #### set next boundary
                if t_geol < 0:
                    model_pars.gridstamp_number = model_pars.gridstamp_number + 1
                    model_pars.next_gridstamp = model_pars.runstamps[model_pars.gridstamp_number]
                else:
                    model_pars.finishgrid = 1

    ######################################################################
    #############   Record plotting states only in sensanal   ############
    ######################################################################

    if sensanal.key == 1:

        workingstate.BAS_AREA.append(BAS_AREA)
        workingstate.GRAN_AREA.append(GRAN_AREA)
        workingstate.DEGASS.append(DEGASS)
        workingstate.delta_mccb.append(delta_mccb)
        workingstate.d34s_S.append(d34s_S)
        workingstate.delta_OSr.append(delta_OSr)
        workingstate.SmM.append(28*S/pars.S0)
        workingstate.CO2ppm.append(RCO2*280)
        workingstate.mrO2.append(mrO2)
        workingstate.iceline.append(iceline)
        workingstate.T_gast.append(TEMP_gast - 273)
        workingstate.ANOX.append(ANOX)
        workingstate.P.append(P/pars.P0)
        workingstate.N.append(N/pars.N0)
        workingstate.time_myr.append(t_geol)
        workingstate.time.append(t)


    ######################################################################
    #########################   Final actions   ##########################
    ######################################################################

    ##### output timestep if specified
    if sensanal.key == 0:
        if pars.telltime == 1:
            if np.mod(stepnumber.step, model_pars.display_resolution) == 0:
                ### print model state to screen
                print('Model step: %d \t time: %s \t next keyframe: %d \n' % (stepnumber.step, t_geol, next_stamp))
    #### record current model step
    stepnumber.step = stepnumber.step + 1


    #### option to bail out if model is running aground
    if stepnumber.step > model_pars.bailnumber:
        return dy

    return dy
