############################################################################################
########## pySCION - Spatial Continuous Integration ##########################################
########## Earth Evolution Model ###########################################################
############################################################################################
#### Coded by BJW Mills
#### b.mills@leeds.ac.uk
####
#### model sensitivity analysis initialiser

###### number of runs

import numpy as np
import SCION_classes
import SCION_initialise
import multiprocessing
from scipy.interpolate import interp1d
import SCION_plot_sens
import time
import matplotlib.pyplot as plt

def SCION_sens(sensruns, no_of_processes):

    sensruns = sensruns
    singlerun = 1
    #multi-core?
    output = []

    if __name__ == "__main__":
        p = multiprocessing.Pool(processes=no_of_processes)
    #should run SCION_initialise with an S=1
        for ind,iteration in enumerate(p.imap_unordered(SCION_initialise.SCION_initialise,
                                          [singlerun]*sensruns)):
            print('iteration:', ind)
            output.append(iteration)

        #make class to store results
        sens = SCION_classes.Sens_class()
        #plots onto a grid of same spacing as first run
        tgrid = output[0].state.time
        #plots onto a regular grid, however sometimes misses small, abrupt changes
        #tgrid = np.arange(output[0].state.time[0], output[0].state.time[-1], 1e6)

        #loop through output and put into class
        for ind, i in enumerate(output):
            interp_BAS_AREA = interp1d(i.state.time,i.state.BAS_AREA.ravel())(tgrid)
            interp_GRAN_AREA = interp1d(i.state.time,i.state.GRAN_AREA.ravel())(tgrid)
            interp_DEGASS = interp1d(i.state.time,i.state.DEGASS.ravel())(tgrid)
            interp_delta_mccb = interp1d(i.state.time,i.state.delta_mccb.ravel())(tgrid)
            interp_d34s_S = interp1d(i.state.time,i.state.d34s_S.ravel())(tgrid)
            interp_delta_OSr = interp1d(i.state.time,i.state.delta_OSr.ravel())(tgrid)
            interp_SmM = interp1d(i.state.time,i.state.SmM.ravel())(tgrid)
            interp_CO2ppm = interp1d(i.state.time,i.state.CO2ppm.ravel())(tgrid)
            interp_mrO2 = interp1d(i.state.time,i.state.mrO2.ravel())(tgrid)
            interp_iceline = interp1d(i.state.time,i.state.iceline.ravel())(tgrid)
            interp_T_gast = interp1d(i.state.time,i.state.T_gast.ravel())(tgrid)
            interp_ANOX = interp1d(i.state.time,i.state.ANOX.ravel())(tgrid)
            interp_P = interp1d(i.state.time,i.state.P.ravel())(tgrid)
            interp_N = interp1d(i.state.time,i.state.N.ravel())(tgrid)
            interp_time_myr = interp1d(i.state.time,i.state.time_myr.ravel())(tgrid)
            interp_time = interp1d(i.state.time,i.state.time.ravel())(tgrid)

            new_data = [interp_BAS_AREA, interp_GRAN_AREA, interp_DEGASS, interp_delta_mccb,
                        interp_d34s_S, interp_delta_OSr, interp_SmM, interp_CO2ppm,
                        interp_mrO2, interp_iceline, interp_T_gast, interp_ANOX, interp_P,
                        interp_N, interp_time_myr, interp_time]

            sens.add_states(new_data)

        ###### plotting
        SCION_plot_sens.SCION_plot_sens(sens)

        return sens, output

##call pySCION sens
sens = SCION_sens(1000, 6)
#
##save
import pickle
filename = 'DEGASS_ONLY_iter=1000'
t = time.localtime()
timestamp = time.strftime('%Y%b%d', t)
with open('./results/model_results/%s_%s.obj' % (filename, timestamp), 'wb') as file_:
    pickle.dump(sens, file_)
