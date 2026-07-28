import numpy as np
import healpy as hp
from astropy.io import fits
from szifi import params, sed
import h5py

class experiment:

    def __init__(self,experiment_name,params_szifi=params.params_szifi_default):

        self.experiment_name = experiment_name
        self.params_szifi = params_szifi
        self.transmission_list = None

        if self.experiment_name == "Planck_real":

            self.nu_eff = np.array([101.31,142.710,221.915,361.290,557.53,862.68])*1e9
            self.noise_levels = np.array([1.29,0.55,0.78,2.56,0.78/1000.*1.7508e4,0.72/1000.*6.9653e5])*60. #all in muK arcmin

            self.FWHM = np.array([9.68,7.3,5.02,4.94,4.83,4.64]) #FWHM1 of https://www.aanda.org/articles/aa/pdf/2016/10/aa25820-15.pdf
            beams_file = fits.open(params_szifi["path"] + "data/HFI_RIMO_Beams-075pc_R2.00.fits")
            indices_beams = [3,4,5,6,11,12]

            self.beams = []

            for i in range(0,len(self.nu_eff)):

                self.beams.append(beams_file[indices_beams[i]].data[0][0])

            self.tsz_sed_paper = np.array([1./(-0.24815),1./(-0.35923),1./5.152,1./0.161098,1./0.06918,1./0.038])*1e6 #tSZ signature in muK

            tsz_sed_model = sed.tsz_model()
            self.tsz_sed = tsz_sed_model.get_sed_exp_bandpass(self)
            self.bandpass_file = fits.open(params_szifi["path"] + "data/HFI_RIMO_Beams-075pc_R2.00.fits")

        if self.experiment_name == "Planck_simple":

            self.nu_eff = np.array([100.,143.,217.,353.,545.,857.])*1e9
            self.FWHM = np.array([9.68,7.3,5.02,4.94,4.83,4.64]) #FWHM1 of https://www.aanda.org/articles/aa/pdf/2016/10/aa25820-15.pdf
            self.noise_levels = np.array([1.29,0.55,0.78,2.56,0.78/1000.*1.7508e4,0.72/1000.*6.9653e5])*60.
            self.MJysr2muK = np.array([4.1877e3,2.632e3,2.0676e3,3.371e3,1.7508e4,6.9653e5])
            self.tsz_sed = np.array([-4.1103e6,-2.8355e6,-2.1188e4,6.1071e6,1.5257e7,3.0228e7])

        elif self.experiment_name == "SPTpol_simple":

            self.nu_eff = np.array([93.,145])*1e9
            self.FWHM = np.array([1.7,1.2])

            self.area_patches = np.array([276.,250.,275.2,251.8,272.9,248.8,277,250.3,274.3,270.8]) #from https://arxiv.org/pdf/1910.04121.pdf
            self.sigma95 = np.array([61.3,59.4,80.4,61.5,54.6,43.8,57.0,54.8,77.6,50.7])
            self.sigma150 = np.array([30.5,36.6,39.2,36.6,28.6,25.3,31.4,31.6,40.,30.])
            self.noise_levels = np.array([np.average(self.sigma95,weights=self.area_patches),np.average(self.sigma150,weights=self.area_patches)])

        elif self.experiment_name == "ACT_simple": #as used in Hilton et al. 2020. Why not 220 GHz?

            self.nu_eff = np.array([100.,145.])*1e9
            self.FWHM = np.array([2.2,1.4])
            self.noise_levels = np.array([33.,27.]) #Guess, based on Fig. 14 of https://arxiv.org/pdf/2007.07290.pdf

        elif self.experiment_name == "ACT_DR6": #as used in Hilton et al. 2020. Why not 220 GHz?

            self.nu_eff = np.array([90.,150.,220])*1e9
            self.FWHM = np.array([1.42, 2.07, 1.01])
            self.noise_levels = np.array([8.4, 8.9, 43]) 

            self.beams = []
            beam_pa_freqs = [[5, "f090"],[6, "f150"],[4, "f220"]]
            for i in range(0,len(self.nu_eff)):
                pa, freq = beam_pa_freqs[i]
                beamfile = "/shared_home/Ziyang/data/ACT_DR6_repeat/beams/beam_"+freq+"_tform.txt"
                d = np.loadtxt(beamfile)
                self.beams.append(d[:,1]/d[0,1])
                # self.beams.append(beams_file[indices_beams[i]].data[0][0])
            # https://arxiv.org/pdf/2503.14451v1
            self.tsz_sed = np.array([-4.191936e6,-2.6336e6,3.9226e5]) #from ACT bandpass 

        elif self.experiment_name == "ACT_DR6_2_freq": #as used in Hilton et al. 2020. Why not 220 GHz?

            self.nu_eff = np.array([90.,150.])*1e9
            self.FWHM = np.array([1.42, 2.07, ])
            self.noise_levels = np.array([8.4, 8.9,]) 

            self.beams = []
            beam_pa_freqs = [[5, "f090"],[6, "f150"],[4, "f220"]]
            for i in range(2):
                pa, freq = beam_pa_freqs[i]
                beamfile = "/shared_home/Ziyang/data/ACT_DR6_repeat/beams/beam_"+freq+"_tform.txt"
                d = np.loadtxt(beamfile)
                self.beams.append(d[:,1]/d[0,1])
                # self.beams.append(beams_file[indices_beams[i]].data[0][0])
            # https://arxiv.org/pdf/2503.14451v1
            self.tsz_sed = np.array([-4.191936e6,-2.6336e6]) #from ACT bandpass 


        elif self.experiment_name == "SO_iso_deep56":
            self.nu_eff = np.array([93., 145])*1e9
            self.tsz_sed = np.array([-4281889.241778137, -2745933.338139288]) # np.array([-4.2840e6, -2.7685e6])
            self.FWHM = np.array([2.2,1.4])
            self.noise_levels = np.array([12, 14]) # inpaint

            freqs = ["f090", "f150"]
            self.beams = []
            for i in range(len(freqs)):
                beamfile_iso = params_szifi["path_data"]+"beams/beam_allarrays_"+freqs[i]+"_tform.txt"
                #Beam_f"+freqs[i]+"_1756699200_20000000000_Interp_Extended_f"+freqs[i]+"_all_tform_format.txt"
                d = np.loadtxt(beamfile_iso)
                self.beams.append(d[:, 1]/d[0, 1])
                
        # elif self.experiment_name == "SO_iso_deep56_tube":
        #     self.nu_eff = np.array([93., 145])*1e9
        #     self.tsz_sed = np.array([-4281889.241778137, -2745933.338139288]) # np.array([-4.2840e6, -2.7685e6])
        #     self.FWHM = np.array([2.2, 1.4])
        #     self.noise_levels = np.array([12, 14]) # inpaint

        #     freqs = ["f090", "f150"]
        #     self.beams = []
        #     for i in range(len(freqs)):
        #         beamfile_iso = params_szifi["path_data"]+"beams/beam_"+params_szifi["map_name"]+"_cmb_tform.txt"
        #         d = np.loadtxt(beamfile_iso)
        #         self.beams.append(d[:, 1]/d[0, 1])
        elif self.experiment_name == "SO_iso_deep56_tube_f090":
            self.nu_eff = np.array([93.])*1e9
            self.tsz_sed = np.array([-4281889.241778137]) # np.array([-4.2840e6, -2.7685e6])
            self.FWHM = np.array([2.2,])
            self.noise_levels = np.array([12,]) # inpaint

            freqs = ["f090"]
            self.beams = []
            for i in range(len(freqs)):
                beamfile_iso = params_szifi["path_data"]+"beams/beam_"+params_szifi["map_name"]+"_cmb_tform.txt"
                d = np.loadtxt(beamfile_iso)
                self.beams.append(d[:, 1]/d[0, 1])
        elif self.experiment_name == "SO_iso_deep56_tube_f150":
            self.nu_eff = np.array([145])*1e9
            self.tsz_sed = np.array([ -2745933.338139288]) # np.array([-4.2840e6, -2.7685e6])
            self.FWHM = np.array([1.4])
            self.noise_levels = np.array([14]) # inpaint

            freqs = ["f150"]
            self.beams = []
            for i in range(len(freqs)):
                beamfile_iso = params_szifi["path_data"]+"beams/beam_"+params_szifi["map_name"]+"_cmb_tform.txt"
                d = np.loadtxt(beamfile_iso)
                self.beams.append(d[:, 1]/d[0, 1])

        elif self.experiment_name == "SO_iso_deep56_beam2":
            print("\n\nSO_iso_deep56_beam2\n\n")
            self.nu_eff = np.array([93., 145])*1e9
            self.tsz_sed = np.array([-4281889.241778137, -2745933.338139288]) # np.array([-4.2840e6, -2.7685e6])
            self.FWHM = np.array([2.2,1.4])
            self.noise_levels = np.array([12, 14]) # inpaint

            freqs = ["f090", "f150"]
            paths = ["/shared_home/Ziyang/data/LAT_iso_deep56/another_beam_saianeesh/profiles_v1r1/data/band/f090/\
window_f090_1756699200_20000000000.txt", "/shared_home/Ziyang/data/LAT_iso_deep56/another_beam_saianeesh/profiles_v1r1/data/band/f150/\
window_f150_1756699200_20000000000.txt"]
            self.beams =[]
            
            for i in range(len(freqs)):
                beamfile_iso =paths[i]
                #Beam_f"+freqs[i]+"_1756699200_20000000000_Interp_Extended_f"+freqs[i]+"_all_tform_format.txt"
                d = np.loadtxt(beamfile_iso)
                self.beams.append(d[:, 1]/d[0, 1])
            

        elif self.experiment_name == "SObaseline_simple": #from https://arxiv.org/pdf/1808.07445.pdf

            # self.nu_eff = np.array([27.,39.,93.,145.,225.,278.])*1e9
            # self.FWHM = np.array([7.4,5.1,2.2,1.4,1.,0.9])
            # self.noise_levels = np.array([71.,36.,8.,10.,22.,54.])
            # self.MJysr2muK = np.array([4.1877e3,2.632e3,2.0676e3,3.371e3,1.7508e4,6.9653e5])
            # self.tsz_sed = np.array([-5.3487e6,-5.2384e6,-4.2840e6,-2.7685e6,3.1517e5,2.7314e6])

            self.nu_eff = np.array([93.,145.,225.,278.])*1e9
            self.FWHM = np.array([2.2,1.4,1.,0.9])
            self.noise_levels = np.array([8.,10.,22.,54.])
            self.MJysr2muK = np.array([2.0676e3,3.371e3,1.7508e4,6.9653e5])
            self.tsz_sed = np.array([-4.2840e6,-2.7685e6,3.1517e5,2.7314e6])
            
        # elif self.experiment_name == "SO_iso_f090":
        #     self.nu_eff = np.array([93.])*1e9
        #     self.tsz_sed = np.array([-4.2840e6])
        #     self.FWHM = np.array([2.2])
        #     self.noise_levels = np.array([25]) 

        #     beamfile_iso = "/shared_home/Ziyang/data/LAT_iso/beam_30_gauss_tform.txt"
        #     self.beams = [np.loadtxt(beamfile_iso)[:, 1]]

        # elif self.experiment_name == "SO_iso_deep56":
        #     self.nu_eff = np.array([93.,145.,225.,278.])*1e9
        #     self.FWHM = np.array([2.2,1.4,1.,0.9])
        #     #not sure
        #     self.noise_levels = np.array([12., 15., 33., 81.])
        #     self.MJysr2muK = np.array([2.0676e3,3.371e3,1.7508e4,6.9653e5])
        #     self.tsz_sed = np.array([-4.2840e6,-2.7685e6,3.1517e5,2.7314e6])

        #     for i in range(0,len(self.nu_eff)):
        #         pa, freq = beam_pa_freqs[i]
        #         beamfile = "/shared_home/Ziyang/data/ACT_DR6_repeat/beams/beam_"+freq+"_tform.txt"
        #         d = np.loadtxt(beamfile)
        #         self.beams.append(d[:,1]/d[0,1])

        #     beamfile_iso = "/shared_home/Ziyang/data/LAT_iso/beam_30_gauss_tform.txt"
        #     self.beams = [np.loadtxt(beamfile_iso)[:, 1]]

            

        elif self.experiment_name == "SOgoal_simple": #from https://arxiv.org/pdf/1808.07445.pdf

            self.nu_eff = np.array([27.,39.,93.,145.,225.,278.])*1e9
            self.FWHM = np.array([7.4,5.1,2.2,1.4,1.,0.9])
            self.noise_levels = np.array([52.,27.,5.8,6.3,15.,37.])
            self.tsz_sed = np.array([-5.3487e6,-5.2384e6,-4.2840e6,-2.7685e6,3.1517e5,2.7314e6])

        self.n_freqs = len(self.nu_eff)

    #Only if beam in self.beam is wanted

    def get_beam(self,i,ptf=True,lmax=4000,nside=2048):

        ell = np.arange(0,lmax+1)
        beam = self.beams[i]

        if ptf == True:

            beam = beam*hp.sphtfunc.pixwin(nside)[0:4001]

        return ell,beam

    def get_ptf(self,nside):

        ptf = hp.sphtfunc.pixwin(nside)
        ell = np.arange(0,len(ptf))

        return ell,ptf

    def get_band_transmission(self):

        if self.experiment_name == "Planck_real":

            bandpass_file = fits.open(self.params_szifi["path"] + "data/HFI_RIMO_R3.00.fits")
            channel_indices = [3,4,5,6,7,8]

            self.transmission_list = []
            self.nu_transmission_list = [] #in Hz

            for i in range(0,len(channel_indices)):

                data_vector = bandpass_file[channel_indices[i]].data

                len_data = len(data_vector)
                wavelength = np.zeros(len_data)
                transmission = np.zeros(len_data)

                for j in range(0,len_data):

                    wavelength[j] = data_vector[j][0]
                    transmission[j] = data_vector[j][1]

                c_light = 299792458.
                nu = wavelength*1e2*c_light

                self.transmission_list.append(transmission[1:])
                self.nu_transmission_list.append(nu[1:])
                
        if self.experiment_name == "ACT_DR6":
            bandpass_file = "/shared_home/Ziyang/data/ACT_DR6_repeat/ACT_DR6_maps/bandpass/AdvACT_Passbands.h5"
            band_names = [90, 150, 220]
            self.transmission_list = []
            self.nu_transmission_list = [] #in Hz
            with h5py.File(bandpass_file, "r") as f:
                for i in range(len(band_names)):
                    if band_names[i] == 90:
                        integration_bounds = f['PA5_f090']["integration_bounds"][:]
                        frequencies = f['PA5_f090']["frequencies"][:]
                        mean_band = f['PA6_f090']["mean_band"][:]
                        
                    if band_names[i] == 150:
                        integration_bounds = f['PA5_f150']["integration_bounds"][:]
                        frequencies = f['PA5_f150']["frequencies"][:]
                        mean_band = f['PA6_f150']["mean_band"][:]
                        
                    if band_names[i] == 220:
                        integration_bounds = f['PA4_f220']["integration_bounds"][:]
                        frequencies = f['PA4_f220']["frequencies"][:]
                        mean_band = f['PA4_f220']["mean_band"][:]
                        
                    indice  = np.where((frequencies>integration_bounds[0])*(frequencies<integration_bounds[1]))
                
                    self.nu_transmission_list.append(frequencies[indice]*1e9) #GHz
                    self.transmission_list.append(mean_band[indice])
                    # print(frequencies, mean_band)
                    # print(frequencies[indice], mean_band[indice])
                    
                                                      
                

class websky_conversions:

    def __init__(self):

        self.freqs = np.array([27.,39.,93.,100.,143.,145.,217.,225.,280.,353.,545.,857.])*1e9
        self.MJysr2muK = np.array([4.5495e4,2.2253e4,4.6831e3,4.1877e3,2.6320e3,2.5947e3,2.0676e3,2.0716e3,2.3302e3,3.3710e3,1.7508e4,6.9653e5])
        self.y2muK = np.array([-5.3487e6,-5.2384e6,-4.2840e6,-4.1103e6,-2.8355e6,-2.7685e6,-2.1188e4,3.1517e5,2.7314e6,6.1071e6,1.5257e7,3.0228e7])

    def get_MJysr2muK(self,freq):

        return np.interp(freq,self.freqs,self.MJysr2muK)

    def get_y2muK(self,freq):

        return np.interp(freq,self.freqs,self.y2muK)
