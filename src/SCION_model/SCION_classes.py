#define classes for pySCION
import numpy as np
import metpy
import metpy.calc as mpcalc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class Variable_parameters_class(object):
    '''
    Class to hold variable parameters used in the SCION calculation/.
    Each of the arguments should be a float or int.
    '''
    def __init__(self, telltime, k_reductant_input, k_mocb, k_locb, k_ocdeg,
    k_ccdeg, k_carbw, k_sfw, basfrac, k_mpsb, k_mgsb, k_pyrw, k_gypw, k_pyrdeg,
    k_gypdeg, k_capb, k_fepb, k_mopb, k_phosw, k_landfrac, k_nfix, k_denit,
    k_Sr_sedw, k_Sr_mantle, k_Sr_silw, k_Sr_metam, k_oxfrac, newp0, copsek16,
    a, b, kfire, P0, O0, A0, G0, C0, PYR0, GYP0, S0, CAL0, N0, OSr0, SSr0,
    suture_factor, arc_factor, relict_arc_factor, root_depth_factor, PALAEOGEOG_TEST,
    BIO_TEST, DEGASSING_TEST, ARC_TEST, SUTURE_TEST):
        #time
        self.telltime = telltime
        #reductant
        self.k_reductant_input  = k_reductant_input
        #organic C cycle
        self.k_locb, self.k_mocb, self.k_ocdeg = k_mocb, k_locb, k_ocdeg
        #carb C cycle
        self.k_ccdeg, self.k_carbw, self.k_sfw, self.basfrac = k_ccdeg, k_carbw, k_sfw, basfrac
        #carb C cycle dependent
        self.k_mccb = self.k_carbw + self.k_ccdeg - self.k_sfw
        self.k_silw = self.k_mccb - self.k_carbw
        self.k_granw = self.k_silw * (1-self.basfrac)
        self.k_basw = self.k_silw * self.basfrac
        #S cycle
        self.k_mpsb, self.k_mgsb, self.k_pyrw, self.k_gypw, self.k_pyrdeg, self.k_gypdeg = k_mpsb, k_mgsb, k_pyrw, k_gypw, k_pyrdeg, k_gypdeg
        #P cycle
        self.k_capb, self.k_fepb, self.k_mopb, self.k_phosw, self.k_landfrac = k_capb, k_fepb, k_mopb, k_phosw, k_landfrac
        #N cycle
        self.k_nfix, self.k_denit = k_nfix, k_denit
        #flux for steady state
        self.k_oxidw = self.k_mocb + self.k_locb - self.k_ocdeg - self.k_reductant_input
        #Sr cycle
        self.k_Sr_sedw, self.k_Sr_mantle, self.k_Sr_silw, self.k_Sr_metam = k_Sr_sedw, k_Sr_mantle, k_Sr_silw, k_Sr_metam
        #Sr cycle dependent
        self.k_Sr_granw = self.k_Sr_silw * (1 - self.basfrac)
        self.k_Sr_basw = self.k_Sr_silw * self.basfrac
        self.total_Sr_removal = self.k_Sr_granw + self.k_Sr_basw + self.k_Sr_sedw + self.k_Sr_mantle
        self.k_Sr_sfw = self.total_Sr_removal * ( self.k_sfw / (self.k_sfw + self.k_mccb) )
        self.k_Sr_sedb = self.total_Sr_removal * ( self.k_mccb / (self.k_sfw + self.k_mccb) )
        #others?
        self.k_oxfrac, self.newp0 = k_oxfrac, newp0
        #COPSE constant for calculating pO2 from normalised O2
        self.copsek16 = copsek16
        #oxidative weathering dependency on O2 concentration
        self.a = a
        #marine organic carbon burial dependency on new production
        self.b = b
        #fire feedback
        self.kfire = kfire
        #resevoir present_ day
        self.P0, self.O0, self.A0, self.G0, self.C0, self.PYR0, self.GYP0 = P0, O0, A0, G0, C0, PYR0, GYP0
        self.S0, self.CAL0, self.N0, self.OSr0, self.SSr0 = S0, CAL0, N0, OSr0, SSr0
        self.suture_factor, self.arc_factor, self.relict_arc_factor = suture_factor, arc_factor, relict_arc_factor
        self.root_depth_factor = root_depth_factor
        self.PGEOG_test = PALAEOGEOG_TEST
        self.BIO_test = BIO_TEST
        self.SUTURE_test = SUTURE_TEST
        self.ARC_test = ARC_TEST
        self.DEGASS_test = DEGASSING_TEST

    def get_CW_present(self, CO2ppm_present_day, INTERPSTACK, k_erosion, Xm, K,
                        kw, Ea, z, sigplus1, T0, R, GRID_AREA_km2):
        #### get present day CO2
        key_upper_CO2_present = np.min(INTERPSTACK.CO2[(INTERPSTACK.CO2 - CO2ppm_present_day) >= 0])
        key_lower_CO2_present = np.max(INTERPSTACK.CO2[(INTERPSTACK.CO2 - CO2ppm_present_day) <= 0])

        #### find keyframe indexes and fractional contribution for present day
        key_upper_CO2_index_present = np.argwhere(INTERPSTACK.CO2 == key_upper_CO2_present)[0][0]
        key_lower_CO2_index_present = np.argwhere(INTERPSTACK.CO2 == key_lower_CO2_present)[0][0]

        #### fractional contribution of each keyframe at present day
        #if dist_to_upper_present + dist_to_lower_present == 0:
        contribution_lower_present = 1
        contribution_upper_present = 0

        #present day runoff
        RUNOFF_present = contribution_upper_present * np.copy(INTERPSTACK.runoff[:,:,key_upper_CO2_index_present,21]) + \
                         contribution_lower_present * np.copy(INTERPSTACK.runoff[:,:,key_lower_CO2_index_present,21])

        #### Tair
        Tair_present = contribution_upper_present * np.copy(INTERPSTACK.Tair[:,:,key_upper_CO2_index_present,21]) + \
                       contribution_lower_present * np.copy(INTERPSTACK.Tair[:,:,key_lower_CO2_index_present,21])
        T_present = Tair_present + 273
        #get slope
        tslope_present = np.copy(INTERPSTACK.slope[:,:,21])#*25
        #### arcs and arc mask
        ARC_present = np.copy(INTERPSTACK.arc[:,:,21])
        arc_mask_present = ARC_present != 0
        #### Sutures and suture mask
        SUTURE_present = np.copy(INTERPSTACK.suture[:,:,21])
        suture_mask_present = SUTURE_present != 0
        ###relict arcs and relict arcs masks
        RELICT_present = np.copy(INTERPSTACK.relict_arc[:,:,21])
        relict_mask_present = RELICT_present != 0
        ###root depth
        ROOT_PRESENCE_present = np.copy(INTERPSTACK.root_presence[:,:,21])
        root_presence_mask_present = ROOT_PRESENCE_present != 0

        #### runoff in mm/yr present day
        Q_present = np.copy(RUNOFF_present)
        Q_present[Q_present<0] = 0

        EPSILON_present = k_erosion * (Q_present**0.31) * tslope_present * np.maximum(Tair_present,2)

        #### equations
        R_T_present = np.exp((Ea / (R * T0)) - (Ea / (R * T_present)))
        R_Q_present = 1 - np.exp(-1 * kw * Q_present)
        R_reg_past = ((z / EPSILON_present)**sigplus1) / sigplus1

        #base chemical weathering
        CW_per_km2_present_raw = 1e6 * EPSILON_present * Xm * (1 - np.exp(-1 * K * R_Q_present * R_T_present * R_reg_past))
        #arc weathering only
        CW_per_km2_present_raw_AF = CW_per_km2_present_raw * (1 + ARC_present * (self.arc_factor - 1)) * arc_mask_present
        #suture weathering only
        CW_per_km2_present_raw_SF = CW_per_km2_present_raw *( 1 + SUTURE_present * ( self.suture_factor - 1 ) ) * suture_mask_present
        #relict arc weathering only
        CW_per_km2_present_raw_RAF = CW_per_km2_present_raw *( 1 + RELICT_present * ( self.relict_arc_factor - 1 ) ) * relict_mask_present
        #root depth weathering only
        CW_per_km2_present_raw_ROOTS = CW_per_km2_present_raw *( 1 + ROOT_PRESENCE_present * ( self.root_depth_factor(0) - 1 ) ) * root_presence_mask_present
        print(np.nansum(CW_per_km2_present_raw_ROOTS), np.nansum(CW_per_km2_present_raw))
        #non arc and suture weathering only
        #check this, might be underestimating
        non_enhanced_weathering_present = CW_per_km2_present_raw * (arc_mask_present != True) * (suture_mask_present != True) * (relict_mask_present != True) * (root_presence_mask_present != True)
        #all weathering
        CW_per_km2_present = CW_per_km2_present_raw_SF + CW_per_km2_present_raw_AF + CW_per_km2_present_raw_RAF + CW_per_km2_present_raw_ROOTS + non_enhanced_weathering_present 

        #print('non_enhanced_weathering_present', np.nansum(non_enhanced_weathering_present*GRID_AREA_km2))
        #print('CW_per_km2_present_raw_ROOTS', np.nansum(CW_per_km2_present_raw_ROOTS*GRID_AREA_km2))
        #print('CW_per_km2_present_raw', np.nansum(CW_per_km2_present_raw*GRID_AREA_km2))
        #print(np.nansum(CW_per_km2_present_raw*GRID_AREA_km2) - np.nansum(CW_per_km2_present_raw_ROOTS*GRID_AREA_km2))
        #print('CW_per_km2_present', np.nansum(CW_per_km2_present*GRID_AREA_km2))


        #### CW total
        CW_present = CW_per_km2_present * GRID_AREA_km2

        #### world CW
        CW_present[np.isnan(CW_present)==1] = 0

        self.CW_present = sum(sum(CW_present))#4.5e8 for palaeogeog=1

    def __bool__ (self):
        return bool(self.telltime)

class Erosion_parameters_class(object):
    '''
    Class to hold erosion parameters used in the SCION calculation/.
    Each of the arguments should be a float or int.
    '''
    def __init__(self, Xm, K, kw, Ea, z, sigplus1, T0, R):
        self.Xm = Xm
        self.K = K
        self.kw = kw
        self.Ea = Ea
        self.z = z
        self.sigplus1 = sigplus1
        self.T0 = T0
        self.R = R

class Model_parameters_class(object):
    '''
    Class to store parameters pertaining to the boundary conditions of the model.
    '''
    def __init__(self, whenstart, whenend, interpstack_time, gridstamp_number, finishgrid, bailnumber,
                 display_resolution, output_length):

        self.whenstart, self.whenend = whenstart, whenend
        self.gridstamp_number, self.finishgrid = gridstamp_number, finishgrid
        self.bailnumber, self.display_resolution = bailnumber, display_resolution
        self.runstamps = interpstack_time[interpstack_time > (self.whenstart * 1e-6)]
        self.next_gridstamp = self.runstamps[0]
        self.output_length = 0 #will update at the end

    def get_rel_contrib(self, interpstack_lat, interpstack_lon):
        lat_areas = np.cos(interpstack_lat * (np.pi/180))
        self.rel_contrib = np.zeros((len(lat_areas),len(interpstack_lon)))
        for ind, lon in enumerate(interpstack_lon):
            self.rel_contrib[:,ind] = lat_areas / np.mean(lat_areas)

    def __bool__ (self):
        return bool(self.whenstart)

class Starting_parameters_class(object):
    '''
    Class to specfically store the starting conditions of the SCION model that are solved each step.
    These form an array of size (21,) and are passed directly to the ODE.
    '''

    def __init__(self, pstart, tempstart, CAL_start, N_start, OSr_start, SSr_start, delta_A_start,
                 delta_S_start, delta_G_start, delta_C_start, delta_PYR_start, delta_GYP_start,
                 delta_OSr_start, delta_SSr_start,ostart, astart, sstart, gstart, cstart,
                 pyrstart,gypstart):

        self.pstart, self.tempstart, self.CAL_start, self.N_start, self.OSr_start = pstart, tempstart, \
                                                                                    CAL_start, N_start, \
                                                                                    OSr_start
        self.SSr_start, self.delta_A_start, self.delta_S_start, self.delta_G_start = SSr_start, delta_A_start, \
                                                                                     delta_S_start, delta_G_start
        self.delta_C_start, self.delta_PYR_start, self.delta_GYP_start = delta_C_start, delta_PYR_start, \
                                                                         delta_GYP_start
        self.delta_OSr_start, self.delta_SSr_start = delta_OSr_start, delta_SSr_start
        self.ostart, self.astart, self.sstart, self.gstart, self.cstart = ostart, astart, sstart, gstart, \
                                                                          cstart
        self.pyrstart, self.gypstart = pyrstart, gypstart
        #actual input parameters, will be [21,] matrix
        self.startstate = np.zeros(21,)
        self.startstate[0] = self.pstart
        self.startstate[1] = self.ostart
        self.startstate[2] = self.astart
        self.startstate[3] = self.sstart
        self.startstate[4] = self.gstart
        self.startstate[5] = self.cstart
        self.startstate[6] = self.pyrstart
        self.startstate[7] = self.gypstart
        self.startstate[8] = self.tempstart
        self.startstate[9] = self.CAL_start
        self.startstate[10] = self.N_start
        self.startstate[11] = self.gstart * self.delta_G_start
        self.startstate[12] = self.cstart * self.delta_C_start
        self.startstate[13] = self.pyrstart * self.delta_PYR_start
        self.startstate[14] = self.gypstart * self.delta_GYP_start
        self.startstate[15] = self.astart * self.delta_A_start
        self.startstate[16] = self.sstart * self.delta_S_start
        self.startstate[17] = self.OSr_start
        self.startstate[18] = self.OSr_start * self.delta_OSr_start
        self.startstate[19] = self.SSr_start
        self.startstate[20] = self.SSr_start * self.delta_SSr_start

    def __bool__ (self):
        return bool(self.whenstart)

class State_class(object):
    '''
    Class for final states (i.e. results)
    '''
    def __init__(self, workingstate, correct_indices):
        #time
        self.time_myr, self.time = workingstate.time_myr[correct_indices], workingstate.time[correct_indices]
        #mass conservation (resevoirs)
        self.iso_res_C, self.iso_res_S, self.res_C, self.res_S = workingstate.iso_res_C[correct_indices], workingstate.iso_res_S[correct_indices],workingstate.res_C[correct_indices], workingstate.res_S[correct_indices]
        #basalt and granite temp dependency
        self.temperature, self.tempC = workingstate.temperature[correct_indices], workingstate.tempC[correct_indices]
        #element resevoirs
        self.P, self.O, self.A, self.S, self.G, self.C, self.N = workingstate.P[correct_indices], workingstate.O[correct_indices], workingstate.A[correct_indices], workingstate.S[correct_indices], workingstate.G[correct_indices], workingstate.C[correct_indices], workingstate.N[correct_indices]
        #mineral resevoirs
        self.PYR, self.GYP, self.OSr, self.SSr = workingstate.PYR[correct_indices], workingstate.GYP[correct_indices], workingstate.OSr[correct_indices], workingstate.SSr[correct_indices]
        #isotope resevoirs1
        self.d13c_A, self.delta_mccb, self.d34s_S, self.delta_G = workingstate.d13c_A[correct_indices], workingstate.delta_mccb[correct_indices], workingstate.d34s_S[correct_indices], workingstate.delta_G[correct_indices]
        #isotope resevoirs2
        self.delta_C, self.delta_PYR, self.delta_GYP, self.delta_OSr = workingstate.delta_C[correct_indices], workingstate.delta_PYR[correct_indices], workingstate.delta_GYP[correct_indices], workingstate.delta_OSr[correct_indices]
        #forcings1
        self.DEGASS, self.W, self.EVO, self.CPLAND = workingstate.DEGASS[correct_indices], workingstate.W[correct_indices], workingstate.EVO[correct_indices], workingstate.CPLAND[correct_indices]
        #forcings2
        self.Bforcing, self.BAS_AREA, self.GRAN_AREA = workingstate.Bforcing[correct_indices], workingstate.BAS_AREA[correct_indices], workingstate.GRAN_AREA[correct_indices]
        #variables
        self.RCO2, self.RO2, self.mrO2, self.VEG, self.ANOX, self.iceline = workingstate.RCO2[correct_indices], workingstate.RO2[correct_indices], workingstate.mrO2[correct_indices], workingstate.VEG[correct_indices], workingstate.ANOX[correct_indices], workingstate.iceline[correct_indices]
        #fluxes1
        self.mocb, self.locb, self.mccb, self.mpsb, self.mgsb, self.silw = workingstate.mocb[correct_indices], workingstate.locb[correct_indices], workingstate.mccb[correct_indices], workingstate.mpsb[correct_indices], workingstate.mgsb[correct_indices], workingstate.silw[correct_indices]
        #fluxes2
        self.carbw, self.oxidw, self.basw, self.granw, self.phosw, self.psea = workingstate.carbw[correct_indices], workingstate.oxidw[correct_indices], workingstate.basw[correct_indices], workingstate.granw[correct_indices], workingstate.phosw[correct_indices], workingstate.psea[correct_indices]
        #fluxes3
        self.nfix, self.denit, self.VEG, self.pyrw, self.gypw, self.ocdeg = workingstate.nfix[correct_indices], workingstate.denit[correct_indices], workingstate.VEG[correct_indices], workingstate.pyrw[correct_indices], workingstate.gypw[correct_indices], workingstate.ocdeg[correct_indices]
        #fluxes4
        self.ccdeg, self.pyrdeg, self.gypdeg, self.sfw, self.Sr_granw, self.Sr_basw = workingstate.ccdeg[correct_indices], workingstate.pyrdeg[correct_indices], workingstate.gypdeg[correct_indices], workingstate.sfw[correct_indices], workingstate.sfw[correct_indices], workingstate.Sr_granw[correct_indices]
        self.Sr_sedw, self.Sr_mantle, self.dSSr, self.relativenewp, self.erosion_tot = workingstate.Sr_sedw[correct_indices], workingstate.Sr_mantle[correct_indices], workingstate.dSSr[correct_indices], workingstate.relativenewp[correct_indices], workingstate.erosion_tot[correct_indices]

class State_class_sensanal(object):
    '''
    Class for final states (i.e. results) if doing sensitvity analysis
    '''
    def __init__(self, workingstate, correct_indices):
        self.BAS_AREA = workingstate.BAS_AREA[correct_indices]
        self.GRAN_AREA = workingstate.GRAN_AREA[correct_indices]
        self.DEGASS = workingstate.DEGASS[correct_indices]
        self.delta_mccb = workingstate.delta_mccb[correct_indices]
        self.d34s_S = workingstate.d34s_S[correct_indices]
        self.delta_OSr = workingstate.delta_OSr[correct_indices]
        self.SmM = workingstate.SmM[correct_indices]
        self.CO2ppm = workingstate.CO2ppm[correct_indices]
        self.mrO2 = workingstate.mrO2[correct_indices]
        self.iceline = workingstate.iceline[correct_indices]
        self.T_gast = workingstate.T_gast[correct_indices]
        self.ANOX = workingstate.ANOX[correct_indices]
        self.P = workingstate.P[correct_indices]
        self.N = workingstate.N[correct_indices]
        self.time_myr = workingstate.time_myr[correct_indices]
        self.time = workingstate.time[correct_indices]

class Workingstate_class(object):
    '''
    Class for storing workingstates as the ODE progresses.
    '''
    def __init__(self):
        #time
        self.time_myr, self.time = [], []
        #mass conservation (resevoirs)
        self.iso_res_C, self.iso_res_S, self.res_C, self.res_S = [], [], [], []
        #basalt and granite temp dependency
        self.temperature, self.tempC = [], []
        #element resevoirs
        self.P, self.O, self.A, self.S, self.G, self.C, self.N = [], [], [], [], [], [], []
        #mineral resevoirs
        self.PYR, self.GYP, self.OSr, self.SSr = [], [], [], []
        #isotope resevoirs1
        self.d13c_A, self.delta_mccb, self.d34s_S, self.delta_G = [], [], [], []
        #isotope resevoirs2
        self.delta_C, self.delta_PYR, self.delta_GYP, self.delta_OSr = [], [], [], []
        #forcings1
        self.DEGASS, self.W, self.EVO, self.CPLAND = [], [], [], []
        #forcings2
        self.Bforcing, self.BAS_AREA, self.GRAN_AREA = [], [], []
        #variables
        self.RCO2, self.RO2, self.mrO2, self.VEG, self.ANOX, self.iceline = [], [], [], [], [], []
        #fluxes1
        self.mocb, self.locb, self.mccb, self.mpsb, self.mgsb, self.silw = [], [], [], [], [], []
        #fluxes2
        self.carbw, self.oxidw, self.basw, self.granw, self.phosw, self.psea = [], [], [], [], [], []
        #fluxes3
        self.nfix, self.denit, self.VEG, self.pyrw, self.gypw, self.ocdeg = [], [], [], [], [], []
        #fluxes4
        self.ccdeg, self.pyrdeg, self.gypdeg, self.sfw, self.Sr_granw, self.Sr_basw = [], [], [], [], [], []
        self.Sr_sedw, self.Sr_mantle, self.dSSr, self.relativenewp, self.erosion_tot = [], [], [], [], []

    def add_workingstates(self, new_data):
        #new data is a list of all the data generated in a solver run
        self.iso_res_C.append(new_data[0]); self.iso_res_S.append(new_data[1]); self.res_C.append(new_data[2]);
        self.res_S.append(new_data[3]); self.time.append(new_data[4]); self.temperature.append(new_data[5]);
        self.tempC.append(new_data[6]); self.P.append(new_data[7]); self.O.append(new_data[8]);
        self.A.append(new_data[9]); self.S.append(new_data[10]); self.G.append(new_data[11]);
        self.C.append(new_data[12]); self.PYR.append(new_data[13]); self.GYP.append(new_data[14]);
        self.N.append(new_data[15]); self.OSr.append(new_data[16]); self.SSr.append(new_data[17]);
        self.d13c_A.append(new_data[18]); self.delta_mccb.append(new_data[19]); self.d34s_S.append(new_data[20]);
        self.delta_G.append(new_data[21]); self.delta_C.append(new_data[22]); self.delta_PYR.append(new_data[23]);
        self.delta_GYP.append(new_data[24]); self.delta_OSr.append(new_data[25]); self.DEGASS.append(new_data[26]);
        self.W.append(new_data[27]); self.EVO.append(new_data[28]); self.CPLAND.append(new_data[29]);
        self.Bforcing.append(new_data[30]);self.BAS_AREA.append(new_data[31]); self.GRAN_AREA.append(new_data[32]);
        self.RCO2.append(new_data[33]); self.RO2.append(new_data[34]); self.mrO2.append(new_data[35]);
        self.VEG.append(new_data[36]); self.ANOX.append(new_data[37]); self.iceline.append(new_data[38]);
        self.mocb.append(new_data[39]); self.locb.append(new_data[40]); self.mccb.append(new_data[41]);
        self.mpsb.append(new_data[42]); self.mgsb.append(new_data[43]); self.silw.append(new_data[44]);
        self.carbw.append(new_data[45]); self.oxidw.append(new_data[46]); self.basw.append(new_data[47]);
        self.granw.append(new_data[48]); self.phosw.append(new_data[49]); self.psea.append(new_data[50]);
        self.nfix.append(new_data[51]); self.denit.append(new_data[52]); self.VEG.append(new_data[53]);
        self.pyrw.append(new_data[54]); self.gypw.append(new_data[55]); self.ocdeg.append(new_data[56]);
        self.ccdeg.append(new_data[57]); self.pyrdeg.append(new_data[58]); self.gypdeg.append(new_data[59]);
        self.sfw.append(new_data[60]); self.Sr_granw.append(new_data[61]); self.Sr_basw.append(new_data[62]);
        self.Sr_sedw.append(new_data[63]); self.Sr_mantle.append(new_data[64]); self.dSSr.append(new_data[65]);
        self.relativenewp.append(new_data[66]); self.erosion_tot.append(new_data[67]); self.time_myr.append(new_data[68])

    def convert_to_array(self):
        self.iso_res_C = np.asarray(self.iso_res_C); self.iso_res_S = np.asarray(self.iso_res_S); self.res_C = np.asarray(self.res_C);
        self.res_S = np.asarray(self.res_S); self.time = np.asarray(self.time); self.temperature = np.asarray(self.temperature);
        self.tempC = np.asarray(self.tempC); self.P = np.asarray(self.P); self.O = np.asarray(self.O);
        self.A = np.asarray(self.A); self.S = np.asarray(self.S); self.G = np.asarray(self.G);
        self.C = np.asarray(self.C); self.PYR = np.asarray(self.PYR); self.GYP = np.asarray(self.GYP);
        self.N = np.asarray(self.N); self.OSr = np.asarray(self.OSr); self.SSr = np.asarray(self.SSr);
        self.d13c_A = np.asarray(self.d13c_A); self.delta_mccb = np.asarray(self.delta_mccb); self.d34s_S = np.asarray(self.d34s_S);
        self.delta_G = np.asarray(self.delta_G); self.delta_C = np.asarray(self.delta_C); self.delta_PYR = np.asarray(self.delta_PYR);
        self.delta_GYP = np.asarray(self.delta_GYP); self.delta_OSr = np.asarray(self.delta_OSr); self.DEGASS = np.asarray(self.DEGASS);
        self.W = np.asarray(self.W); self.EVO = np.asarray(self.EVO); self.CPLAND = np.asarray(self.CPLAND);
        self.Bforcing = np.asarray(self.Bforcing);self.BAS_AREA = np.asarray(self.BAS_AREA); self.GRAN_AREA = np.asarray(self.GRAN_AREA);
        self.RCO2 = np.asarray(self.RCO2); self.RO2 = np.asarray(self.RO2); self.mrO2 = np.asarray(self.mrO2);
        self.VEG = np.asarray(self.VEG); self.ANOX = np.asarray(self.ANOX); self.iceline = np.asarray(self.iceline);
        self.mocb = np.asarray(self.mocb); self.locb = np.asarray(self.locb); self.mccb = np.asarray(self.mccb);
        self.mpsb = np.asarray(self.mpsb); self.mgsb = np.asarray(self.mgsb); self.silw = np.asarray(self.silw);
        self.carbw = np.asarray(self.carbw); self.oxidw = np.asarray(self.oxidw); self.basw = np.asarray(self.basw);
        self.granw = np.asarray(self.granw); self.phosw = np.asarray(self.phosw); self.psea = np.asarray(self.psea);
        self.nfix = np.asarray(self.nfix); self.denit = np.asarray(self.denit); self.VEG = np.asarray(self.VEG);
        self.pyrw = np.asarray(self.pyrw); self.gypw = np.asarray(self.gypw); self.ocdeg = np.asarray(self.ocdeg);
        self.ccdeg = np.asarray(self.ccdeg); self.pyrdeg = np.asarray(self.pyrdeg); self.gypdeg = np.asarray(self.gypdeg);
        self.sfw = np.asarray(self.sfw); self.Sr_granw = np.asarray(self.Sr_granw); self.Sr_basw = np.asarray(self.Sr_basw);
        self.Sr_sedw = np.asarray(self.Sr_sedw); self.Sr_mantle = np.asarray(self.Sr_mantle); self.dSSr = np.asarray(self.dSSr);
        self.relativenewp = np.asarray(self.relativenewp); self.erosion_tot = np.asarray(self.erosion_tot); self.time_myr = np.asarray(self.time_myr)

class Workingstate_class_sensanal(object):
    '''
    Class for storing workingstates as the ODE progresses.
    '''
    def __init__(self):
        #time
        self.BAS_AREA = []
        self.GRAN_AREA = []
        self.DEGASS = []
        self.delta_mccb = []
        self.d34s_S = []
        self.delta_OSr = []
        self.SmM = []
        self.CO2ppm = []
        self.mrO2 = []
        self.iceline = []
        self.T_gast = []
        self.ANOX = []
        self.P = []
        self.N = []
        self.time_myr = []
        self.time = []

    def convert_to_array(self):
        self.BAS_AREA = np.asarray(self.BAS_AREA)
        self.GRAN_AREA = np.asarray(self.GRAN_AREA)
        self.DEGASS = np.asarray(self.DEGASS)
        self.delta_mccb = np.asarray(self.delta_mccb)
        self.d34s_S = np.asarray(self.d34s_S)
        self.delta_OSr = np.asarray(self.delta_OSr)
        self.SmM = np.asarray(self.SmM)
        self.CO2ppm = np.asarray(self.CO2ppm)
        self.mrO2 = np.asarray(self.mrO2)
        self.iceline = np.asarray(self.iceline)
        self.T_gast = np.asarray(self.T_gast)
        self.ANOX = np.asarray(self.ANOX)
        self.P = np.asarray(self.P)
        self.N = np.asarray(self.N)
        self.time_myr = np.asarray(self.time_myr)
        self.time = np.asarray(self.time)

class Run_class(object):

    def __init__(self, state, gridstate, pars, model_pars, start_pars, forcings,
                 erosion_pars):
        self.state = state
        self.gridstate = gridstate
        self.pars = pars
        self.model_pars = model_pars
        self.start_pars = start_pars
        self.forcings = forcings
        self.erosion_pars = erosion_pars

class Run_class_sensanal(object):

    def __init__(self):
        self.state = []
        self.gridstate = []
        self.pars = []
        self.model_pars = []
        self.start_pars = []
        self.forcings = []

class Interpstack_class(object):

    def __init__(self, CO2, time, Tair, runoff, land, lat, lon, topo, aire,
                 gridarea, suture, arc, relict_arc, slope, root_presence):

        self.CO2, self.time, self.Tair, self.runoff, self.land = CO2.astype(float), time, Tair.astype(float), runoff.astype(float), land
        self.lat, self.lon, self.topo, self.aire, self.gridarea = lat, lon, topo.astype(float), aire.astype(float), gridarea
        self.suture, self.arc, self.relict_arc = suture, arc, relict_arc
        self.slope = slope
        self.root_presence = root_presence

    def get_masks(self):

        self.arc_mask = self.arc != 0
        #... and no sutures
        self.suture_mask = self.suture != 0
        #... and no relict arcs
        self.relict_arc_mask = self.relict_arc != 0
        #... and no relict arcs
        self.root_presence_mask = self.root_presence != 0

    def get_enhancements(self, pars):

        self.arc_enhancement = ( 1 + self.arc * ( pars.arc_factor - 1 ) ) * self.arc_mask
        self.suture_enhancement = ( 1 + self.suture * ( pars.suture_factor - 1 ) ) * self.suture_mask
        self.relict_arc_enhancement = ( 1 + self.relict_arc * ( pars.relict_arc_factor - 1 ) ) * self.relict_arc_mask
        self.root_enhancement = ( 1 + self.root_presence * ( pars.root_depth_factor(self.time) - 1 ) ) * self.root_presence_mask

    def __bool__ (self):
        return bool(self.telltime)

class Forcings_class(object):

    def __init__(self, t, B, BA, Ca, CP, D, E, GA, PG, U, W, coal, epsilon,
                 GR_BA_df, GA_df, forcing_degassing, forcing_shoreline):

        self.t, self.B, self.BA, self.Ca, self.CP, self.D = t, B, BA, Ca, CP, D
        self.E, self.GA, self.PG, self.U, self.W = E, GA, PG, U, W
        self.coal, self.epsilon = coal, epsilon

        self.GR_BA = np.asarray([GR_BA_df['t'].to_numpy()*1e6,
                                 GR_BA_df['BA'].to_numpy()])
        self.newGA = np.asarray([GA_df['t'].to_numpy()*1e6,
                                 GA_df['GA'].to_numpy()])

        #lots of degassing, default SCION is D_force_mid/min/max
        #Merdith curves from sub zones Merdith et al. 2021
        #Marcilly curves from Marcilly et al. 2021
        self.D_force_x = forcing_degassing['D_force_x'][0] #time
        #default SCION
        #self.D_force_mid = forcing_degassing['D_force_mid'].reshape(len(forcing_degassing['D_force_mid']),) #reshape to (601,)
        #self.D_force_min = forcing_degassing['D_force_min'].reshape(len(forcing_degassing['D_force_min']),) #reshape to (601,)
        #self.D_force_max = forcing_degassing['D_force_max'].reshape(len(forcing_degassing['D_force_max']),) #reshape to (601,)
        #compiled curve v1, no rift modifier pre 450 Ma
        #self.D_force_COMPLETE_min_SMOOTH = forcing_degassing['degass_COMPLETE_min_SMOOTH_v1'].reshape(len(forcing_degassing['degass_COMPLETE_min_SMOOTH_v1']),) #reshape to (601,)
        #self.D_force_COMPLETE_max_SMOOTH = forcing_degassing['degass_COMPLETE_max_SMOOTH_v1'].reshape(len(forcing_degassing['degass_COMPLETE_max_SMOOTH_v1']),) #reshape to (601,)
        #self.D_force_COMPLETE_mean_SMOOTH = forcing_degassing['degass_COMPLETE_mean_SMOOTH_v1'].reshape(len(forcing_degassing['degass_COMPLETE_mean_SMOOTH_v1']),) #reshape to (601,)
        #compiled curve v2,  conservative rifts, fisher diffuse, cao arcs
        #self.D_force_COMPLETE_min_SMOOTH = forcing_degassing['degass_COMPLETE_min_SMOOTH_v2'].reshape(len(forcing_degassing['degass_COMPLETE_min_SMOOTH_v2']),) #reshape to (601,)
        #self.D_force_COMPLETE_max_SMOOTH = forcing_degassing['degass_COMPLETE_max_SMOOTH_v2'].reshape(len(forcing_degassing['degass_COMPLETE_max_SMOOTH_v2']),) #reshape to (601,)
        #self.D_force_COMPLETE_mean_SMOOTH = forcing_degassing['degass_COMPLETE_mean_SMOOTH_v2'].reshape(len(forcing_degassing['degass_COMPLETE_mean_SMOOTH_v2']),) #reshape to (601,)
        #compiled curve v3, conservative rifts with modifier pre 450 Ma, Cao et al. arcs min MER21 et al arcs max
        #self.D_force_COMPLETE_min_SMOOTH = forcing_degassing['degass_COMPLETE_min_SMOOTH_v3'].reshape(len(forcing_degassing['degass_COMPLETE_min_SMOOTH_v3']),) #reshape to (601,)
        #self.D_force_COMPLETE_max_SMOOTH = forcing_degassing['degass_COMPLETE_max_SMOOTH_v3'].reshape(len(forcing_degassing['degass_COMPLETE_max_SMOOTH_v3']),) #reshape to (601,)
        #self.D_force_COMPLETE_mean_SMOOTH = forcing_degassing['degass_COMPLETE_mean_SMOOTH_v3'].reshape(len(forcing_degassing['degass_COMPLETE_mean_SMOOTH_v3']),) #reshape to (601,)
        #compiled curve v4, no rifts, Fischer diffuse, Merdith arcs
        #self.D_force_COMPLETE_min_SMOOTH = forcing_degassing['degass_COMPLETE_min_SMOOTH_v4'].reshape(len(forcing_degassing['degass_COMPLETE_min_SMOOTH_v4']),) #reshape to (601,)
        #self.D_force_COMPLETE_max_SMOOTH = forcing_degassing['degass_COMPLETE_max_SMOOTH_v4'].reshape(len(forcing_degassing['degass_COMPLETE_max_SMOOTH_v4']),) #reshape to (601,)
        #self.D_force_COMPLETE_mean_SMOOTH = forcing_degassing['degass_COMPLETE_mean_SMOOTH_v4'].reshape(len(forcing_degassing['degass_COMPLETE_mean_SMOOTH_v4']),) #reshape to (601,)
        #compiled curve v5, mean rifts with modifier pre 450 Ma, Fischer diffuse, Merdith arcs
        #self.D_force_COMPLETE_min_SMOOTH = forcing_degassing['degass_COMPLETE_min_SMOOTH_v5'].reshape(len(forcing_degassing['degass_COMPLETE_min_SMOOTH_v5']),) #reshape to (601,)
        #self.D_force_COMPLETE_max_SMOOTH = forcing_degassing['degass_COMPLETE_max_SMOOTH_v5'].reshape(len(forcing_degassing['degass_COMPLETE_max_SMOOTH_v5']),) #reshape to (601,)
        #self.D_force_COMPLETE_mean_SMOOTH = forcing_degassing['degass_COMPLETE_mean_SMOOTH_v5'].reshape(len(forcing_degassing['degass_COMPLETE_mean_SMOOTH_v5']),) #reshape to (601,)
        #compiled curve v6, conservative rifts with modifier pre 450 Ma, Fischer diffuse, group min/max for arcs
        self.D_force_COMPLETE_min_SMOOTH = forcing_degassing['degass_COMPLETE_min_SMOOTH_v6'].reshape(len(forcing_degassing['degass_COMPLETE_min_SMOOTH_v6']),) #reshape to (601,)
        self.D_force_COMPLETE_max_SMOOTH = forcing_degassing['degass_COMPLETE_max_SMOOTH_v6'].reshape(len(forcing_degassing['degass_COMPLETE_max_SMOOTH_v6']),) #reshape to (601,)
        self.D_force_COMPLETE_mean_SMOOTH = forcing_degassing['degass_COMPLETE_mean_SMOOTH_v6'].reshape(len(forcing_degassing['degass_COMPLETE_mean_SMOOTH_v6']),) #reshape to (601,)
		
        self.shoreline_time = forcing_shoreline['shoreline_time'][0]
        self.shoreline_relative = forcing_shoreline['shoreline_relative'][0]

    def get_interp_forcings(self):

        self.E_reloaded_INTERP = interp1d(1e6 * self.t, self.E)
        self.W_reloaded_INTERP = interp1d(1e6 * self.t, self.W)
        self.GR_BA_reloaded_INTERP = interp1d(self.GR_BA[0], self.GR_BA[1])
        self.newGA_reloaded_INTERP = interp1d(self.newGA[0], self.newGA[1])
        #mid point [solo study] degass curves
        self.D_complete_SMOOTH_INTERP = interp1d(self.D_force_x, self.D_force_COMPLETE_mean_SMOOTH)
        self.D_complete_min_SMOOTH_INTERP = interp1d(self.D_force_x, self.D_force_COMPLETE_min_SMOOTH)
        self.D_complete_max_SMOOTH_INTERP = interp1d(self.D_force_x, self.D_force_COMPLETE_max_SMOOTH)

        self.shoreline_INTERP = interp1d(self.shoreline_time, self.shoreline_relative)
        self.f_biot_INTERP = interp1d([-1000e6, -525e6, -520e6, 0],[0, 0, 1, 1])
        self.CB_INTERP = interp1d([0, 1], [1.2, 1])      

    def __bool__ (self):
        return bool(self.telltime)

class Stepnumber_class(object):

    def __init__(self, step):
        self.step = step

class Gridstate_class(object):
    def __init__(self, gridstate_array):
        self.time_myr = np.copy(gridstate_array)
        self.land = np.copy(gridstate_array)
        self.SUTURE = np.copy(gridstate_array)
        self.ARC = np.copy(gridstate_array)
        self.RELICT_ARC = np.copy(gridstate_array)
        self.ROOT_DEPTH = np.copy(gridstate_array)
        self.Q = np.copy(gridstate_array)
        self.Tair = np.copy(gridstate_array)
        self.TOPO = np.copy(gridstate_array)
        self.CW = np.copy(gridstate_array)
        self.CWcarb = np.copy(gridstate_array)
        self.EPSILON = np.copy(gridstate_array)

class Sensanal_class(object):
    def __init__(self, sensanal_key):
        self.key = sensanal_key

    def __bool__ (self):
        return bool(self.key)

class Plotrun_class(object):
    def __init__(self, plotrun_key):
        self.key = plotrun_key

    def __bool__ (self):
        return bool(self.key)

class Gtune_class(object):
    def __init__(self, gtune_key):
        self.key = gtune_key

    def __bool__ (self):
        return bool(self.key)

class Sensparams_class(object):

    def __init__(self, randminusplus1, randminusplus2, randminusplus3,
                 randminusplus4, randminusplus5, randminusplus6, randminusplus7):
        self.randminusplus1 = randminusplus1
        self.randminusplus2 = randminusplus2
        self.randminusplus3 = randminusplus3
        self.randminusplus4 = randminusplus4
        self.randminusplus5 = randminusplus5
        self.randminusplus6 = randminusplus6
        self.randminusplus7 = randminusplus7

class Sens_class(object):
    '''
    for storing
    '''
    def __init__(self):
        self.BAS_AREA = []
        self.GRAN_AREA = []
        self.DEGASS = []
        self.delta_mccb = []
        self.d34s_S = []
        self.delta_OSr = []
        self.SmM = []
        self.CO2ppm = []
        self.mrO2 = []
        self.iceline = []
        self.T_gast = []
        self.ANOX = []
        self.P = []
        self.N = []
        self.time_myr = []
        self.time = []

    def add_states(self, new_data):
        self.BAS_AREA.append(new_data[0])
        self.GRAN_AREA.append(new_data[1])
        self.DEGASS.append(new_data[2])
        self.delta_mccb.append(new_data[3])
        self.d34s_S.append(new_data[4])
        self.delta_OSr.append(new_data[5])
        self.SmM.append(new_data[6])
        self.CO2ppm.append(new_data[7])
        self.mrO2.append(new_data[8])
        self.iceline.append(new_data[9])
        self.T_gast.append(new_data[10])
        self.ANOX.append(new_data[11])
        self.P.append(new_data[12])
        self.N.append(new_data[13])
        self.time_myr.append(new_data[14])
        self.time.append(new_data[15])

    def convert_to_array(self):
        self.BAS_AREA = np.asarray(self.BAS_AREA)
        self.GRAN_AREA = np.asarray(self.GRAN_AREA)
        self.DEGASS = np.asarray(self.DEGASS)
        self.delta_mccb = np.asarray(self.delta_mccb)
        self.d34s_S = np.asarray(self.d34s_S)
        self.delta_OSr = np.asarray(self.delta_OSr)
        self.SmM = np.asarray(self.SmM)
        self.CO2ppm = np.asarray(self.CO2ppm)
        self.mrO2 = np.asarray(self.mrO2)
        self.iceline = np.asarray(self.iceline)
        self.T_gast = np.asarray(self.T_gast)
        self.ANOX = np.asarray(self.ANOX)
        self.P = np.asarray(self.P)
        self.N = np.asarray(self.N)
        self.time_myr = np.asarray(self.time_myr)
        self.time = np.asarray(self.time)
