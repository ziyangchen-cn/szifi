import numpy as np
import scipy.signal as sg
from scipy import ndimage
from szifi import params, maps, model, cat, spec, utils, sed, peaks
import warnings, os
warnings.filterwarnings("ignore")
import gzip
from tqdm import tqdm

#Core class for cluster finding
class cluster_finder:

    def __init__(self,params_szifi=params.params_szifi_default,
    params_model=params.params_model_default,
    data_file=None,
    rank=0):

        self.params_szifi = params_szifi
        self.params_model = params_model
        self.data_file = data_file.data #instance of input_data
        self.theta_500_vec = self.params_szifi["theta_500_vec_arcmin"]
        self.rank = rank
        self.exp = self.data_file["experiment"]
        self.cosmology = model.cosmological_model(self.params_szifi).cosmology
        self.indices_filter=None
        self.wcs = None

        if self.params_szifi["detection_method"] == "DBSCAN":

            from sklearn.cluster import DBSCAN

        if self.params_szifi["mmf_type"] == "standard" and self.params_szifi["a_matrix"] is None:

            a_matrix = np.zeros((len(self.exp.tsz_sed),1))
            a_matrix[:,0] = self.exp.tsz_sed
            self.params_szifi["a_matrix"] = a_matrix

        if self.params_szifi["deproject_cib"] is not None:

            get_a_matrix_cib(self.params_szifi,self.params_model,self.data_file)

    def find_clusters(self):

        #Print some information

        if self.params_szifi["mmf_type"] == "standard":

            type_name = "standard"

        elif self.params_szifi["mmf_type"] == "spectrally_constrained":

            n_dep = self.params_szifi["a_matrix"].shape[1]-1
            type_name = "spectrally constrained, " + str(n_dep) + " component(s) deprojected"

        print("SZiFi")
        print("")
        print("")
        print("MMF type:",type_name)
        print("Iterative:",self.params_szifi["iterative"])
        print("Extraction mode:",self.params_szifi["extraction_mode"])
        print("Experiment:",self.exp.experiment_name)
        print("Frequency channels:",self.params_szifi["freqs"])

        #Iterate over fields

        self.results_dict = {}

        for field_id in self.data_file["params_data"]["field_ids"]:

            print("")
            print("")
            print("Field",field_id)
            print("")

            #Gather input data
            ## Note this deletes information from the input object (self.data_file) so it cannot be reused (this is to save memory)

            map_dtype = self.params_szifi["map_dtype"]

            # self.t_obs = utils.extract(self.data_file,"t_obs",field_id,map_dtype) #in muK
            # self.t_noi = utils.extract(self.data_file,"t_noi",field_id,map_dtype) #in muK

            self.t_obs = self.data_file["t_obs"][field_id] #in muK
            self.t_noi = self.data_file["t_noi"][field_id] #in muK 
            
            self.mask_point = utils.extract(self.data_file,"mask_point",field_id,map_dtype)
            # mask_window: apodized galaxy mask used as the window function for power-spectrum
            # /covariance estimation (spec.power_spectrum, cspec.get_inv_cov/get_cov below). Must
            # stay numerically identical to mask_map (same array, written to both keys by the
            # survey file) -- the matched filter is built from this window's noise statistics and
            # then applied to mask_map-windowed data, so the two have to match for the filter to
            # be correctly calibrated. Not the point-source mask (that's mask_point/use_ps_mask).
            self.mask_window = utils.extract(self.data_file,"mask_window", field_id,map_dtype)
            self.mask_select = utils.extract(self.data_file,"mask_select",field_id,map_dtype)
            self.mask_select_no_tile = utils.extract(self.data_file,"mask_select_no_tile",field_id,map_dtype)
            self.mask_select_buffer = utils.extract(self.data_file, "mask_select_buffer", field_id, map_dtype)
            self.mask_map = utils.extract(self.data_file,"mask_map",field_id,map_dtype)
            self.mask_peak_finding_no_tile = utils.extract(self.data_file,"mask_peak_finding_no_tile",field_id,map_dtype)
            self.mask_peak_finding = utils.extract(self.data_file,"mask_peak_finding",field_id,map_dtype)

            self.weights = None

            if self.params_szifi["snr_weighting"] == True:

                self.weights = utils.extract(self.data_file,"snr_weights",field_id,map_dtype)
 
            self.dx = self.data_file["dx_arcmin"][field_id]/60./180.*np.pi
            self.nx = self.data_file["nx"][field_id]
            self.dy = self.dx
            self.ny = self.nx

            if "dy_arcmin" in self.data_file:

                self.dy = self.data_file["dy_arcmin"][field_id]/60./180.*np.pi

            if "ny" in self.data_file:
               
                self.ny = self.data_file["ny"][field_id]

            self.pix = maps.pixel(self.nx,self.dx,ny=self.ny,dy=self.dy)
            
            if self.params_szifi["decouple_type"] == "master" and self.params_szifi["compute_coupling_matrix"] == False:

                self.coupling_matrix_name = self.data_file["coupling_matrix_name"][field_id]

            else:

                self.coupling_matrix_name = None

            if self.params_szifi["extraction_mode"] == "fixed":

                self.catalogue_fixed = self.data_file["catalogue_input"][field_id]
                # self.catalogue_fixed = cat.get_catalogue_indices(self.catalogue_fixed, self.catalogue_fixed.catalogue["m_500"]>5)
                # print(self.catalogue_fixed.get_n_clusters())

            if self.params_szifi["tile_type"] == "car":

                self.wcs = self.t_obs.wcs

            #Select frequency channels to use

            self.t_obs = maps.select_freqs(self.t_obs,self.params_szifi["freqs"])
            self.t_noi = maps.select_freqs(self.t_noi,self.params_szifi["freqs"])
            
            if self.params_szifi["get_q_true"] == True:

                self.t_true = utils.extract(self.data_file,"t_true",field_id,map_dtype)
                self.t_true = maps.select_freqs(self.t_true,self.params_szifi["freqs"])

            
            #Inpaint point sources

            if self.params_szifi["inpaint"] == True:

                
                self.t_obs = maps.inpaint_freq(self.t_obs,
                                               self.mask_point,
                                               n_inpaint=self.params_szifi["n_inpaint"],
                                               pix=self.pix,
                                               noise=self.exp.noise_levels,
                                               inpaint_type=self.params_szifi["inpaint_type"],
                                               )
                
            mask_point_0 = self.mask_point
            
            #Initialise results class

            self.results = cat.results_detection()
            self.results.info = self.params_szifi
            self.results.theta_500_vec = self.theta_500_vec
            self.results.fsky = np.sum(self.mask_select)*self.pix.dx*self.pix.dy/(4.*np.pi)

            #Initialise cluster masking (if iterative noise covariance estimation is required)

            mask_cluster = np.ones((self.pix.nx,self.pix.ny),dtype=self.t_obs.dtype)
            clusters_masked_old = cat.cluster_catalogue()

            #Apodise (except t_noise, which is apodised in the power spectra estimation functions)
            if self.params_szifi["apod_type"] == "new":

                self.t_obs = maps.multiply_t(self.mask_map,self.t_obs)

                if self.params_szifi["get_q_true"] == True:

                    self.t_true = maps.multiply_t(self.mask_map,self.t_true)

            #Filter input maps in harmonic space

            [lmin,lmax] = self.params_szifi["lrange"]

            self.indices_filter = tuple(np.asarray(np.where((maps.rmap(self.pix).get_ell() < lmin) |  (maps.rmap(self.pix).get_ell() > lmax)),dtype=np.int32))
            
            t_noi_original_before_filter = self.t_noi.copy()
            
            self.t_obs = maps.filter_tmap(self.t_obs,self.pix,self.params_szifi["lrange"],indices_filter=self.indices_filter)
            self.t_noi = maps.filter_tmap(self.t_noi,self.pix,self.params_szifi["lrange"],indices_filter=self.indices_filter)
            
            t_noi_original = self.t_noi
            

            if self.params_szifi["get_q_true"] == True:

                self.t_true = maps.filter_tmap(self.t_true,self.pix,self.params_szifi["lrange"],indices_filter=self.indices_filter)

            #Initiate loop over iterative noise covariance estimation

            i = 0

            while i >= 0:

                if self.rank == 0:

                    print("\n\n\n Noise it",i)

                if not np.any(self.mask_select):

                    print(f"{self.rank}: mask_select all zeroes for field {field_id}, iteration {i}; breaking")

                    break

                self.mask_point = mask_point_0*mask_cluster

                #Inpaint noise map (as it may change with iteration)
                
                if self.params_szifi["inpaint"] == True:

                    self.t_noi = maps.inpaint_freq(t_noi_original,
                                                   self.mask_point,
                                                   n_inpaint=self.params_szifi["n_inpaint"],
                                                   pix=self.pix,
                                                   noise=self.exp.noise_levels,
                                                   inpaint_type=self.params_szifi["inpaint_type"])
                    #debug save
                    # if self.params_szifi["path_diagnosis"]:
                    #     np.savez(self.params_szifi["path_diagnosis"]+"inpainted_maps_"+field_id+"_it"+str(i), noise_maps = self.t_noi)
            
            
                    
                                        
                    for ll in range(0,self.t_noi.shape[2]):
            
                        self.t_noi[:,:,ll] = maps.circular_cosine_blend_adaptive(t_noi_original[:,:,ll],self.t_noi[:,:,ll],self.mask_point,inner_zero_frac=0.)

                    self.t_noi = maps.filter_tmap(self.t_noi,self.pix,self.params_szifi["lrange"],indices_filter=self.indices_filter)

                print("it", i, np.std(self.t_noi), np.sum(self.mask_point))
                #Estimate channel cross-spectra

                lmax1d = self.params_szifi["powspec_lmax1d"]
                new_shape = self.params_szifi["powspec_new_shape"]

                if lmax1d is not None:

                    if new_shape is not None:

                        raise ValueError("Only one of powspec_lmax1d or powspec_new_shape can be specified")

                    # new_shape = maps.get_newshape_lmax1d((self.pix.nx,self.pix.ny),lmax1d,self.pix.dx)
                    new_shape = maps.get_newshape_lmax1d_anisotropic((self.pix.nx,self.pix.ny),lmax1d,self.pix.dx, self.pix.dy)
                
                
                bin_fac = self.params_szifi["powspec_bin_fac"]
                
                self.ps = spec.power_spectrum(self.pix,
                mask=self.mask_window,
                cm_compute=self.params_szifi["coupling_matrix_needed"],
                cm_compute_scratch=self.params_szifi["compute_coupling_matrix"],
                cm_save=self.params_szifi["save_coupling_matrix"],
                cm_name=self.coupling_matrix_name,
                bin_fac=bin_fac,
                new_shape=new_shape)

                self.cspec = spec.cross_spec(np.arange(len(self.params_szifi["freqs"])), 
                save_label = "",)

                self.cspec.get_cross_spec(self.pix,
                t_map=self.t_noi,
                ps=self.ps,
                decouple_type=self.params_szifi["decouple_type"],
                inpaint_flag=self.params_szifi["inpaint"],
                mask_point=self.mask_point,
                lsep=self.params_szifi["lsep"],
                bin_fac=bin_fac,
                new_shape=new_shape, 
                )

                
                
#                 print("save", self.params_szifi["path_template"]+"input_"+str(field_id)+"_lmax"+str(lmax1d)+".npz")
#                 np.savez(self.params_szifi["path_template"]+"input_"+str(field_id)+"_lmax"+str(lmax1d)+".npz", pix = self.pix, t_map=self.t_noi, new_shape=new_shape, mask=self.mask_window, ell=self.cspec.ell_vec, spec_tensor=self.cspec.spec_tensor,
# mask_select = self.mask_select, mask_point=self.mask_point, cov = cov)

                # save_name = "/shared_home/Ziyang/Cluster_Finder/realisation_1/cov/cov_"+self.params_szifi["field_id"]+"_R"+str(self.params_szifi["R_mask"])+"_"+self.params_szifi["cov_type"]+".npz"
                
                print("\n", np.std(self.t_noi), "\n")
                self.inv_cov = self.cspec.get_inv_cov(self.pix,
                                                      t_map=self.t_noi,
                                                      interp_type=self.params_szifi["interp_type"],
                                                      bin_fac=bin_fac,
                                                      new_shape=new_shape,
                                                      mask=self.mask_window,
                                                      cov_type=self.params_szifi["cov_type"],
                                                      cov_kernel_shape=self.params_szifi["cov_kernel_shape"])#, save_name =save_name)


                if self.params_szifi["path_diagnosis"]:
                    
                    # cov is just for test
                    cov = self.cspec.get_cov(self.pix,
                                                          t_map=self.t_noi,
                                                          interp_type=self.params_szifi["interp_type"],
                                                          bin_fac=bin_fac,
                                                          new_shape=new_shape,
                                                          mask=self.mask_window,
                                                          cov_type=self.params_szifi["cov_type"],
                                                          cov_kernel_shape=self.params_szifi["cov_kernel_shape"])#, save_name =save_name)
                    np.savez(self.params_szifi["path_diagnosis"]+"cov_"+field_id+"_it"+str(i)+"_apo"+str(self.params_szifi["aposcale"]), cov = cov, t_noi = self.t_noi, pix = self.pix, new_shape=new_shape, mask=self.mask_window, inv_cov = self.inv_cov)

            
                del self.mask_point

                #Compute weights for constrained MMF

                if self.params_szifi["mmf_type"] == "standard":

                    self.params_szifi["cmmf_type"] = "standard_mmf"

                self.cmmf = scmmf_precomputation(pix=self.pix,
                freqs=self.params_szifi["freqs"],
                inv_cov=self.inv_cov,
                lrange=self.params_szifi["lrange"],
                beam_type=self.params_szifi["beam"],
                exp=self.exp,
                cmmf_type=self.params_szifi["cmmf_type"],
                a_matrix=self.params_szifi["a_matrix"],
                comp_to_calculate=self.params_szifi["comp_to_calculate"],
                mmf_type=self.params_szifi["mmf_type"])
                

                tilemask_mode = self.params_szifi["tilemask_mode"]
                if tilemask_mode == "catalogue": # Peak-find on whole field and apply tile mask to catalogues
                    # Includes a "buffer" catalogue to ensure no clusters are missed on the borders of the tiles
                    mask_peak_finding_names_list = ["field"]
                    mask_peak_finding_dict = {"field": self.mask_peak_finding_no_tile, "names": mask_peak_finding_names_list}
                    if i == 0:
                        del self.mask_peak_finding # Not needed
                    mask_select_names_dict = {"field":["field", "tile"]}
                    mask_select_dict = {"field": self.mask_select_no_tile, "tile": self.mask_select,
                                        "names": mask_select_names_dict}
                    catalogue_find_key = ("field", "tile")
                    catalogue_buffer_key = "empty"
                    for_masking_key = ("field", "field")

                elif tilemask_mode == "catalogue_buffer": # Peak-find on whole field and apply tile mask to catalogues

                    # Includes a "buffer" catalogue to ensure no clusters are missed on the borders of the tiles
                    mask_peak_finding_names_list = ["field"]
                    mask_peak_finding_dict = {"field": self.mask_peak_finding_no_tile, "names": mask_peak_finding_names_list}
                    if i == 0:
                        del self.mask_peak_finding # Not needed
                    mask_select_names_dict = {"field":["field", "tile", "buffer"]}
                    mask_select_dict = {"field": self.mask_select_no_tile, "tile": self.mask_select,
                                        "buffer": self.mask_select_buffer, "names": mask_select_names_dict}
                    catalogue_find_key = ("field", "tile")
                    catalogue_buffer_key = ("field", "buffer")
                    for_masking_key = ("field", "field")

                elif tilemask_mode == "field": # Apply tile mask before peak-finding. Legacy mode used for Planck.

                    mask_peak_finding_names_list = ["field", "tile"]
                    mask_peak_finding_dict = {"field": self.mask_peak_finding_no_tile, "tile": self.mask_peak_finding,
                                              "names": mask_peak_finding_names_list}
                    mask_select_names_dict= {"field":["field"], "tile":["tile"]}
                    mask_select_dict = {"field": self.mask_select_no_tile, "tile": self.mask_select, "names": mask_select_names_dict}
                    catalogue_find_key = ("tile", "tile")
                    catalogue_buffer_key = "empty"
                    for_masking_key = ("field", "field")

                else:

                    raise ValueError(f"SziFi parameter tilemask_mode=\"{tilemask_mode}\" invalid. Allowed values are 'catalogue' and 'field'.")

                #Matched filter construction
                self.filtered_maps = filter_maps(t_obs=self.t_obs,
                inv_cov=self.inv_cov,
                noise_type = self.params_szifi["noise_type"],
                pix=self.pix,
                cosmology=self.cosmology,
                theta_500_vec=self.theta_500_vec,
                field_id=field_id,
                i_it=i,
                params=self.params_szifi,
                params_model=self.params_model,
                mask_map=self.mask_map,
                mask_select_dict=mask_select_dict,
                mask_peak_finding_dict=mask_peak_finding_dict,
                rank=self.rank,
                exp=self.exp,
                cmmf=self.cmmf,
                wcs=self.wcs,
                weights=self.weights,
                lmax = lmax)


                # filename = self.params_szifi["path_template"]+"test_"+field_id+".npz"
                # print("save", filename)
                # np.savez(filename, spec_tensor = self.cspec.spec_tensor, ell_vec = self.cspec.ell_vec, cov = cov, t_noi2 = self.t_noi, t_noi1 =t_noi_original_before_filter,
                #         t_obs=self.t_obs, mask_map = self.mask_map)
                

                #SZiFi in cluster finding mode: blind cluster detection
                

                if self.params_szifi["extraction_mode"] == "find" or (self.params_szifi["extraction_mode"] == "fixed" and self.params_szifi["iterative"] == True):

                    if self.rank == 0:

                        print("Cluster finding")
    
                    self.filtered_maps.find_clusters() # main, cluster number
                    
                    
                    self.results.sigma_vec["find_" + str(i)] = self.filtered_maps.sigma_vec
                    self.results.f_sky_apo = self.params_szifi["f_sky_apo"]
                    self.results.tile_frac = self.params_szifi["tile_frac"]

                    self.results.catalogues["catalogue_find_" + str(i)] = self.filtered_maps.results[catalogue_find_key]
                    self.results.catalogues["catalogue_find_buffer_" + str(i)] = self.filtered_maps.results[catalogue_buffer_key]
                    results_for_masking = self.filtered_maps.results[for_masking_key]

                    if self.rank == 0:

                        print("Detections SNR",np.flip(np.sort(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"])))
                        print("theta_500", np.flip(np.sort(self.results.catalogues["catalogue_find_" + str(i)].catalogue["theta_500"])))

                #SZiFi in fixed mode: extraction for an input catalogue

                if (self.params_szifi["extraction_mode"] == "fixed"):
                    
                    con1 = (self.params_szifi["iterative"] == True and i > 0)
                    con2 = (self.params_szifi["iterative"] == False)
                    con3 = (self.params_szifi["iterative"] == True and len(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"]) == 0.)
                    con4 = (self.params_szifi["iterative"] == True and all(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"] < self.params_szifi["q_th_noise"]))
                    

                    if (con1 or con2 or con3 or con4):

                        if self.rank == 0:

                            print("Extraction for input catalogue")

                        # Fixed mode extraction: choose fast interpolation or full computation
                        # Interpolation uses pre-computed q_tensor from find_clusters (4.4x speedup)
                        # Full computation redoes MMF for each cluster (slow but more accurate)
                        if self.params_szifi.get("use_q_tensor_interp", False) and \
                           hasattr(self.filtered_maps, 'q_tensor'):
                            # Fast mode: 3D interpolation on q_tensor from find_clusters
                            catalogue_obs = self.filtered_maps.extract_at_true_values_interp(
                                self.catalogue_fixed,
                                q_tensor=self.filtered_maps.q_tensor,
                                y_tensor=self.filtered_maps.y_tensor,
                                theta_vec=self.theta_500_vec,
                                pix=self.pix,
                                wcs=self.wcs
                            )
                        else:
                            # Original slow method: recompute MMF for each cluster
                            self.filtered_maps.extract_at_true_values(self.t_obs,
                            self.catalogue_fixed)
                            catalogue_obs = self.filtered_maps.catalogue_true_values


                        self.results.catalogues["catalogue_fixed_" + str(i)] = catalogue_obs

                        print("Fixed SNR",np.flip(np.sort(self.results.catalogues["catalogue_fixed_" + str(i)].catalogue["q_opt"])))
                print(self.results.catalogues.keys())

                #Extract true SNR

                if self.params_szifi["get_q_true"] == True:

                    con1 = (self.params_szifi["iterative"] == True and i > 0)
                    con2 = (self.params_szifi["iterative"] == False)
                    con3 = (self.params_szifi["iterative"] == True and len(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"]) == 0.)
                    con4 = (self.params_szifi["iterative"] == True and all(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"] < self.params_szifi["q_th_noise"]))

                    if (con1 or con2 or con3 or con4):

                        self.filtered_maps.extract_at_true_values(self.t_true,
                        self.catalogue_fixed,
                        comp_to_calculate=self.params_szifi["comp_to_calculate"],
                        profile_type=self.params_model["profile_type"])
                        catalogue_true_values = self.filtered_maps.catalogue_true_values

                        self.catalogue_true_values = catalogue_true_values
                        self.results.catalogues["catalogue_true_" + str(i)] = catalogue_true_values

                        print("q true",np.flip(np.sort(self.results.catalogues["catalogue_true_" + str(i)].catalogue["q_opt"])))

                print(494, "it_max", self.params_szifi["max_it"])
                #Several breaking conditions
                if (self.params_szifi["extraction_mode"] == "find"):
                    self.n_clusters = len(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"])  + \
                                      len(self.results.catalogues["catalogue_find_buffer_" + str(i)].catalogue["q_opt"])
                elif (self.params_szifi["extraction_mode"] == "fixed"):
                    # self.n_clusters = len(self.results.catalogues["catalogue_fixed_" + str(i)].catalogue["q_opt"])
                    self.n_clusters = self.catalogue_fixed.get_n_clusters()

                if self.n_clusters == 0.:
                    break

                if self.params_szifi["iterative"] == False:
                    break

                if i == self.params_szifi["max_it"]:
                    break

                #Cluster masking for iterative covariance estimation

                self.results_for_masking = results_for_masking
                
                clusters_masked_new = cat.apply_q_cut(self.results_for_masking,self.params_szifi["q_th_noise"])
                clusters_masked_old = cat.apply_q_cut(clusters_masked_old,self.params_szifi["q_th_noise"])

                if len(clusters_masked_new.catalogue["q_opt"]) == 0:
                    break

                if self.params_szifi["tile_type"] == "healpix":

                    lonlat = False

                elif self.params_szifi["tile_type"] == "car":

                    lonlat = "equatorial"

                clusters_masked_old_id,clusters_masked_new_id = cat.identify_clusters(clusters_masked_old,clusters_masked_new,lonlat=lonlat)


                # if np.all(clusters_masked_old_id.catalogue["q_opt"] != -1) and i > 0:
                #     break

                mask_cluster = get_cluster_mask(self.pix,
                self.results_for_masking,
                self.params_szifi["q_th_noise"],
                self.params_szifi["mask_radius"],
                tile_type=self.params_szifi["tile_type"],
                wcs=self.wcs,
                max_masking_radius_arcmin=self.params_szifi["max_radius_mask_arcmin"],
                min_masking_radius_arcmin=self.params_szifi["min_radius_mask_arcmin"])

                clusters_masked_old = clusters_masked_new

                i += 1

            # Clean up the saved templates

            if self.params_szifi["save_and_load_template"] and hasattr(self,"filtered_maps"):

                for j_theta in range(len(self.theta_500_vec)):
                    break
                    os.remove(self.filtered_maps.template_name % j_theta)
                    os.remove(self.filtered_maps.template_norm_name % j_theta)
            
            if self.params_szifi["tile_type"] == "healpix" and self.params_szifi["get_lonlat"] == True:

                self.results.get_lonlat(field_id,self.pix,nside=self.data_file["nside_tile"])

            self.results_dict[field_id] = self.results



#Get mask at the location of detected clusters (for iterative noise covariance estimation)

def get_cluster_mask(pix,catalogue,q_th_noise,mask_radius,max_masking_radius_arcmin=np.inf,tile_type="healpix",wcs=None,min_masking_radius_arcmin=0.):

    mask_cluster = np.ones((pix.nx,pix.ny))
    

    for j in tqdm(range(0,len(catalogue.catalogue["q_opt"]))):

        if catalogue.catalogue["q_opt"][j] > q_th_noise:
                
            source_coords = np.zeros((1,2))

            if tile_type == "healpix":

                x_est,y_est = maps.get_theta_misc([catalogue.catalogue["theta_x"][j],catalogue.catalogue["theta_y"][j]],pix)
                source_coords[0,0] = x_est
                source_coords[0,1] = y_est

            elif tile_type == "car":

                source_coords[0,0] = catalogue.catalogue["ra"][j]
                source_coords[0,1] = catalogue.catalogue["dec"][j]

            masking_radius_arcmin = max(min_masking_radius_arcmin,min(max_masking_radius_arcmin,catalogue.catalogue["theta_500"][j]*mask_radius))
            mask_cluster *= maps.ps_mask(pix,1,masking_radius_arcmin,tile_type=tile_type,wcs=wcs).get_mask_map(source_coords=source_coords)

    return mask_cluster

#Construct and apply MMF

class filter_maps:

    def __init__(self,t_obs=None,inv_cov=None,pix=None,cosmology=None,theta_500_vec=None,
    params=None,field_id=0,i_it=0,mask_map=None,mask_select_dict=None,
    mask_peak_finding_dict=None,rank=0,exp=None,cmmf=None,params_model=None,indices_filter=None,wcs=None,weights=None, noise_type="", lmax = 0):

        self.t_obs = t_obs
        self.inv_cov = inv_cov
        self.noise_type = noise_type
        self.pix = pix
        self.cosmology = cosmology
        self.theta_500_vec = theta_500_vec
        self.params = params
        self.params_model = params_model
        self.field_id = field_id
        self.i_it = i_it
        self.mask_map = mask_map
        self.mask_select_dict = mask_select_dict
        self.mask_peak_finding_dict = mask_peak_finding_dict
        self.mask_names_finding = mask_peak_finding_dict["names"] # ['peak_finding_name1', 'peak_finding_name2',...]
        self.mask_names_select = mask_select_dict["names"] # {'peak_finding_name':['select_name1',]};
        self.rank = rank
        self.theta_500_vec = theta_500_vec
        self.exp = exp
        self.cmmf = cmmf
        self.indices_filter=indices_filter
        self.wcs = wcs
        self.weights = weights
        self.lmax = lmax

        self.theta_range = [0.,pix.nx*pix.dx,0.,pix.ny*pix.dy]

        if self.params["save_and_load_template"]:
            
            self.template_name = self.params["path_template"] + f"tem_{self.field_id}_%s.npy" # % j (theta index)
            self.template_norm_name = self.params["path_template"] + f"tem_norm_{self.field_id}_%s.npy"

    #Find detections blindly

    def find_clusters(self):

        t_obs = self.t_obs #shape = (nx, ny, len(freq))
        

        if self.params["apod_type"] == "old":
            t_obs = maps.multiply_t(self.mask_map,t_obs)
            t_obs = maps.filter_tmap(t_obs,self.pix,self.params["lrange"],indices_filter=self.indices_filter)
        n_theta = len(self.theta_500_vec)
        # detect_peaks_maxima_lomem = (self.params["detection_method"] == "maxima_lomem") and (n_theta > 3)
        # Low-memory peak finding: only use if NOT in q_tensor interpolation mode
        # maxima_lomem stores only 3 theta layers (rolling window) to save memory
        # BUT q_tensor interpolation needs full (nx, ny, n_theta) array
        # So when use_q_tensor_interp=True, force standard q_tensor storage
        detect_peaks_maxima_lomem = (self.params["detection_method"] == "maxima_lomem") \
                                     and (n_theta > 3) \
                                     and (not self.params.get("use_saddle_consolidation", False)) \
                                     and (not self.params.get("use_q_tensor_interp", False))

        
        if detect_peaks_maxima_lomem:

            self.peak_info = {}

            for mask_name in self.mask_names_finding:

                self.peak_info[mask_name] = np.empty((13,0),dtype=np.float32)

            self.q_tensor = np.zeros((self.pix.nx,self.pix.ny,3),dtype=self.params["map_dtype"])
            self.y_map_last = None

        else:

            self.q_tensor = np.zeros((self.pix.nx,self.pix.ny,n_theta),dtype=self.params["map_dtype"])
            self.y_tensor = np.zeros((self.pix.nx,self.pix.ny,n_theta),dtype=self.params["map_dtype"])

        self.sigma_vec = np.zeros(n_theta)
        # self.sigma_vec = []

        #new: point source
        if self.params_model["profile_type"] == "point":
            point_map = np.zeros((self.pix.ny, self.pix.nx, 1))
            point_map[self.pix.ny//2, self.pix.nx//2, 0] = 1.0 
            t_tem = ps.get_t_map_convolved(self.pix,tile_type=self.params["tile_type"],wcs=self.wcs)
            t_tem_norm = None

            return 0
        
        # 2026-10-26 (opt-in, default off): cross-response Q_SZ(theta_j) = (tem_j . filter_ref)/(tem_ref . filter_ref) of the FIXED theta_ref filter
        # (params 'qsz_ref_ind') to a theta_j cluster, per tile and iteration, saved to params 'qsz_path' as [theta_500_vec, qsz]; standard MMF, arnaud only.
        qsz_ref = self.params.get("qsz_ref_ind", None)
        if qsz_ref is not None:
            jr = int(qsz_ref); nfw_r = model.gnfw(model.get_m_500(self.theta_500_vec[jr], 0.4, self.cosmology), 0.4, self.cosmology, type=self.params_model["profile_type"])
            t_r = nfw_r.get_t_map_convolved(self.pix, self.exp, beam=self.params["beam"], theta_cart=[(0.5*self.pix.nx)*self.pix.dx, (0.5*self.pix.nx)*self.pix.dx],
                get_nc=False, sed=False, eval_type=self.params.get("template_eval_type", "standard"), tile_type=self.params["tile_type"], wcs=self.wcs)
            tem_r = maps.filter_tmap(t_r/nfw_r.get_y_norm(self.params["norm_type"]), self.pix, self.params["lrange"], indices_filter=self.indices_filter)
            filt_r = np.asarray(maps.get_ifft_f(get_tem_conv_fft(self.pix, maps.get_fft_f(tem_r, self.pix), self.inv_cov, mmf_type=self.params["mmf_type"],
                cmmf_prec=self.cmmf, comp=0, field_id=self.field_id), self.pix).real, dtype=t_obs.dtype)
            norm_r = np.sum(maps.get_tmap_times_fvec(tem_r, self.cmmf.a_matrix[:, 0]).real*filt_r)*self.pix.dx*self.pix.dy; qsz = np.zeros(n_theta); del t_r, tem_r

        # MMF
        for j in range(0,n_theta):

            if self.rank == 0:

                print("Theta",j,self.theta_500_vec[j])

            if self.params["save_and_load_template"] == True:

                template_name = self.template_name % j
                template_norm_name = self.template_norm_name % j

                if self.i_it == 0 and os.path.isfile(template_name):

                    raise RuntimeError(f"template file {template_name} exists already; overwriting not allowed to prevent collisions during parallel runs")

            
            if self.params["save_and_load_template"] == False or (self.params["save_and_load_template"] == True and self.i_it == 0):
                
                if self.params_model["profile_type"] == "point": #Todo: CAR functionality

                    ps = model.point_source(self.exp,beam_type=self.params["beam"])
                    t_tem = ps.get_t_map_convolved(self.pix,tile_type=self.params["tile_type"],wcs=self.wcs)
                    t_tem_norm = None

                elif self.params_model["profile_type"] == "arnaud":

                    z = 0.4 #???
                    M_500 = model.get_m_500(self.theta_500_vec[j],z,self.cosmology)
                    nfw = model.gnfw(M_500,z,self.cosmology,
                    type=self.params_model["profile_type"])

                    theta_cart = [(0.5*self.pix.nx)*self.pix.dx,
                    (0.5*self.pix.nx)*self.pix.dx]

                    if self.params["mmf_type"] == "standard" or (self.params["mmf_type"] == "spectrally_constrained" and self.params["cmmf_type"] == "one_dep"):

                        t_tem = nfw.get_t_map_convolved(self.pix,
                        self.exp,
                        beam=self.params["beam"],
                        theta_cart=theta_cart,
                        get_nc=False,
                        sed=False,
                        eval_type=self.params.get("template_eval_type","standard"),
                        tile_type=self.params["tile_type"],
                        wcs=self.wcs
                        )
                        t_tem_norm = None

                    elif self.params["mmf_type"] == "spectrally_constrained" and self.params["cmmf_type"] == "general":

                        t_tem_norm,t_tem = nfw.get_t_map_convolved(self.pix,
                        self.exp,
                        beam=self.params["beam"],
                        theta_cart=theta_cart,
                        get_nc=True,
                        sed=False,
                        eval_type=self.params.get("template_eval_type","standard"),
                        tile_type=self.params["tile_type"],
                        wcs=self.wcs
                        )
                        

                        t_tem_norm = t_tem_norm/nfw.get_y_norm(self.params["norm_type"])
                        t_tem_norm = maps.filter_tmap(t_tem_norm,
                        self.pix,
                        self.params["lrange"],
                        indices_filter=self.indices_filter)

                    t_tem = t_tem/nfw.get_y_norm(self.params["norm_type"])
                
                tem = maps.filter_tmap(t_tem,
                self.pix,
                self.params["lrange"],
                indices_filter=self.indices_filter)
                

                if self.params["save_and_load_template"] == True:

                    np.save(template_name,tem)
                    np.save(template_norm_name,t_tem_norm)

            elif self.params["save_and_load_template"] == True and self.i_it > 0:

                tem = np.load(template_name,allow_pickle=True)[()]
                t_tem_norm = np.load(template_norm_name,allow_pickle=True)[()]

            path_save = ""
            # If user provided a filters output dir in params (minimal per-tile save), use it.
            # The saved file is a small .npz produced inside get_mmf_q_map (filter, tem, pix).
            filters_dir = self.params.get("path_filters", "")
            if filters_dir:
                try:
                    os.makedirs(filters_dir, exist_ok=True)
                except Exception:
                    pass
                path_save = os.path.join(filters_dir, f"filter_{self.field_id}_nth{j}_it{self.i_it}.npz")
            elif self.params["path_diagnosis"] and "n_theta" in self.params:
                if j in self.params["n_theta"]:
                    path_save = self.params["path_diagnosis"]+"filter_"+self.field_id+"_l"+str(self.lmax)+"_nth"+str(j)+"_it"+str(self.i_it)

                
            if qsz_ref is not None:   # before get_mmf_q_map, which multiplies tem by the SED in place
                qsz[j] = np.sum(maps.get_tmap_times_fvec(tem, self.cmmf.a_matrix[:, 0]).real*filt_r)*self.pix.dx*self.pix.dy/norm_r
                if j == n_theta - 1:
                    os.makedirs(self.params["qsz_path"], exist_ok=True)
                    np.save(os.path.join(self.params["qsz_path"], f"qsz_{self.field_id}_it{self.i_it}.npy"), np.array([self.theta_500_vec, qsz]))

            q_map,y_map,std = get_mmf_q_map(t_obs,
            tem,
            self.inv_cov,
            self.pix,
            mmf_type=self.params["mmf_type"],
            cmmf_prec=self.cmmf,
            tem_norm=t_tem_norm, 
             f_sky_apo = self.params["f_sky_apo"],
             field_id = self.field_id,
             noise_type = self.noise_type, path_save = path_save)
            print("it", self.i_it, " std:", std)
            
            #save filter        
            # print("save:", self.params["path_template"]+"qmap_"+self.field_id+"_nth"+str(j)+"_it"+str(self.i_it))
            # np.savez(self.params["path_template"]+"qmap_"+self.field_id+"_nth"+str(j)+"_it"+str(self.i_it), t_obs = t_obs, tem = tem, inv_cov=self.inv_cov, t_tem_norm = t_tem_norm, q_map = q_map, y_map = y_map, noise = std)
            # exit()
        
            if self.params["snr_weighting"] == True:
                
                q_map = q_map*self.weights
                #std = std/weights

            del tem, t_tem_norm

            
            if self.params["save_snr_maps"] == True and (
                self.params.get("snr_maps_theta_ind") is None
                or j in self.params["snr_maps_theta_ind"]):

                fil = gzip.GzipFile(self.params["snr_maps_path"] + "/" + self.params["snr_maps_name"] + "_q_" + str(self.i_it) + "_" + str(j) + ".npy.gz", "w")
                np.save(file=fil,arr=q_map)
                #np.save(file=fil,arr=q_map*self.mask_select_dict["tile"])
                fil.close()

                
            
            if detect_peaks_maxima_lomem:

                # For "maxima_lomem" method we do peak-finding here to save memory

                q_th = self.params["q_th"]

                if j == 0:

                    self.q_tensor[:,:,1] = q_map
                    self.y_map_last = y_map

                else:

                    self.q_tensor[:,:,2] = q_map

                    qm = self.q_tensor[:,:,1] # We actually want to use the previous one

                    # KNOWN ISSUE: clusters are silently dropped when two adjacent theta
                    # layers hold bit-identical q values.  Not fixed here; get_maxima_mask
                    # (used when use_q_tensor_interp is on) has the corrected test.
                    #
                    # Cause: argmax returns the FIRST maximum, so this test requires the
                    # candidate layer to be STRICTLY above both theta neighbours.  The true
                    # maximum of q(theta) almost never sits exactly on a grid layer -- it
                    # falls between two of them, which makes those two layers nearly equal.
                    # Once their difference drops below one float32 ulp they are the same
                    # number, and then BOTH are rejected: layer k loses to k+1 here, and
                    # k+1 loses to k on its own turn.  The detection is not displaced by one
                    # layer, it vanishes from the catalogue entirely, even at high SNR.
                    #
                    # Example (CFC lowmass/mss2, seed 0, tile sq8deg_2_32, it1): q = 16.651106
                    # at theta layers 42 and 43, both 0x41853577, while layer 41 is 866 ulp
                    # away.  Spatially the same voxel led its four neighbours by 0.7-2.2 in q,
                    # so only the theta test failed.
                    #
                    # Rate: finer theta grids bring adjacent layers closer together, so ties
                    # get more frequent roughly as (dlog theta)^-2.  Measured fraction of it0
                    # peaks with q>10 that have no it1 counterpart (seed 0, 450 tiles):
                    #     n_theta =  80 :  0/8085  (0.00%)
                    #     n_theta = 100 :  6/8098  (0.07%)
                    #     n_theta = 120 :  6/8094  (0.07%)
                    #     n_theta = 200 : 25/8133  (0.31%)
                    # Which clusters are hit is effectively random from run to run, because
                    # the it1 noise covariance is re-estimated from a randomly inpainted map
                    # and shifts q by ~1e-4 in relative terms, far more than one ulp.
                    #
                    # Why it is not fixed here: the plateau test in get_maxima_mask needs
                    # layer k+2 to confirm that a tied pair is a genuine maximum rather than
                    # a flat step on a slope, and this rolling window only holds k-1, k, k+1.
                    # Fixing it properly means deepening the window to four layers and
                    # delaying the peak test by two layers instead of one.
                    zmax = np.argmax(self.q_tensor, axis=2) == 1

                    for mask_name in self.mask_names_finding:

                        q_opt,y0_est,maxs_idx, refined = make_detections2(qm,self.y_map_last,self.mask_peak_finding_dict[mask_name],q_th,zmax, self.params)

                        self.save_detections(q_opt, y0_est,maxs_idx,mask_name,j-1, refined)

                    if j == n_theta-1:

                        qm = self.q_tensor[:,:,2]
                        # Same tie issue as above (see the note at the previous zmax).
                        zmax = np.argmax(self.q_tensor[:,:,1:], axis=2) == 1

                        for mask_name in self.mask_names_finding:

                            q_opt,y0_est,maxs_idx, refined = make_detections2(qm,y_map,self.mask_peak_finding_dict[mask_name],q_th,zmax, self.params)
                            self.save_detections(q_opt,y0_est,maxs_idx,mask_name,j, refined)

                    self.q_tensor = np.roll(self.q_tensor,-1,axis=2)
                    self.y_map_last = y_map

                    del qm, zmax, q_map, y_map

            else:

                self.q_tensor[:,:,j] = q_map
                self.y_tensor[:,:,j] = y_map

                del q_map, y_map

            self.sigma_vec[j] = std

        self.results = {}
        
        if self.params["tile_type"] == "healpix":

            x_coord = maps.rmap(self.pix).get_x_coord_map_wrt_origin() #vertical coordinate, in rad
            y_coord = maps.rmap(self.pix).get_y_coord_map_wrt_origin() #horizontal coordinate, in rad

        elif self.params["tile_type"] == "car":

            from pixell import enmap
            position_map = enmap.empty(t_obs[:,:,0].shape,self.wcs)
            dec_map, ra_map = position_map.posmap(corner = True)
            x_coord = ra_map
            y_coord = dec_map
            
        
        for mask_name in self.mask_names_finding:

            if detect_peaks_maxima_lomem:
                q_opt, y0_est, theta_est, inds0, inds1,\
                i_interp, j_interp, q_interp, y0_interp, i_com, j_com, q_com, y0_com = self.peak_info[mask_name]
                inds0, inds1 = inds0.astype(np.int64), inds1.astype(np.int64)
                #print(1);exit()
                x_est = x_coord[(inds0, inds1)]
                y_est = y_coord[(inds0, inds1)]

                refined = {"i_interp": i_interp,"j_interp": j_interp,"q_interp": q_interp,"y0_interp": y0_interp,"i_com": i_com,"j_com": j_com,"q_com": q_com,"y0_com": y0_com,}

                
            else:

                # deepcopy(self.q_tensor) here would be safest, otherwise self.q_tensor is modified in place
                # But for memory savings, this is OK if the order is (no_tile, tile)
                if self.mask_names_finding not in [['field'],['field','tile']]:

                    raise ValueError("mask list must be ['field'] or ['field','tile']")
                
                # import pylab as pl
                # pl.figure()
                # pl.imshow(self.q_tensor[:,:,0]*self.mask_select_dict[self.mask_names_select[mask_name][1]],vmax=5.)
                # pl.savefig("figures/q_map.pdf",origin='lower')
                # pl.show()
                
                q_tensor = apply_mask_peak_finding(self.q_tensor,self.mask_peak_finding_dict[mask_name])
                indices = make_detections(q_tensor,self.params["q_th"],self.pix,detection_method=self.params["detection_method"])
                

                
                q_opt = q_tensor[indices]
                y_tensor = apply_mask_peak_finding(self.y_tensor,self.mask_peak_finding_dict[mask_name])
                y0_est = y_tensor[indices]
                x_est = x_coord[(indices[0],indices[1])]
                y_est = y_coord[(indices[0],indices[1])]
                theta_est = self.theta_500_vec[indices[2]]
                #n_detect = len(q_opt)

                
                # interpolation in blind mode
                peak_i = np.asarray(indices[0], dtype=np.int64)
                peak_j = np.asarray(indices[1], dtype=np.int64)
                theta_idx = np.asarray(indices[2], dtype=np.int64)
                refined = {"i_interp": peak_i.astype(float),
                            "j_interp": peak_j.astype(float),
                            "q_interp": q_opt.copy(),
                            "y0_interp": y0_est.copy(),
                            "i_com": peak_i.astype(float),
                            "j_com": peak_j.astype(float),
                            "q_com": q_opt.copy(),
                            "y0_com": y0_est.copy(),}
                
                for th in np.unique(theta_idx):
                    sel = theta_idx == th
                    ref = refine_detection_positions_2d(
                        q_tensor[:, :, th], y_tensor[:, :, th],
                        peak_i[sel], peak_j[sel],
                        self.params["q_th"], self.mask_peak_finding_dict[mask_name], self.params
                    )
                    for key in refined:
                        refined[key][sel] = ref[key]

            cat_new = cat.cluster_catalogue()

            cat_new.catalogue["q_opt"] = q_opt
            cat_new.catalogue["y0"] = y0_est
            cat_new.catalogue["theta_500"] = theta_est
            cat_new.catalogue["pixel_ids"] = [self.field_id]*len(q_opt)

            print("Cluster number:", len(q_opt))

            cat_new.catalogue["q_opt_interp"] = refined["q_interp"]
            cat_new.catalogue["y0_interp"] = refined["y0_interp"]
            # COM fields disabled (not needed)
            # cat_new.catalogue["q_opt_com"] = refined["q_com"]
            # cat_new.catalogue["y0_com"] = refined["y0_com"]

  
            
            
            if self.params["tile_type"] == "healpix":

                theta_x = y_est + self.theta_range[0]
                theta_y = self.pix.nx*self.pix.dx-x_est + self.theta_range[2]
                cat_new.catalogue["theta_x"] = theta_x
                cat_new.catalogue["theta_y"] = theta_y

            elif self.params["tile_type"] == "car":
                
                cat_new.catalogue["ra"] = x_est
                cat_new.catalogue["dec"] = y_est

                dec_interp, ra_interp = enmap.pix2sky(
                    t_obs[:, :, 0].shape,
                    self.wcs,
                    np.array([refined["i_interp"], refined["j_interp"]]),
                    corner=True,
                )

                # COM sky coordinates disabled (not needed)
                # dec_com, ra_com = enmap.pix2sky(
                #     t_obs[:, :, 0].shape,
                #     self.wcs,
                #     np.array([refined["i_com"], refined["j_com"]]),
                #     corner=True,
                # )

                cat_new.catalogue["ra_interp"] = ra_interp
                cat_new.catalogue["dec_interp"] = dec_interp
                # cat_new.catalogue["ra_com"] = ra_com
                # cat_new.catalogue["dec_com"] = dec_com
                
            
            for mask_name_select in self.mask_names_select[mask_name]:
                if len(cat_new.catalogue["dec"])>0:
                    # new merge algorithm: 2026-06-20
                    if self.params.get("use_saddle_consolidation", False):
                        print("\n use consolidate_saddle_catalogue \n")
                        cat_new = peaks.consolidate_saddle_catalogue(
                            cat_new,
                            q_tensor,                   # line 887
                            self.theta_500_vec,
                            self.wcs,
                            res_arcmin=0.5,             # SO CAR pixel size
                            q_key="q_opt",
                            theta_cap_arcmin=4.0,
                            saddle_depth=2.0,
                            link_radius_arcmin=12.0,
                        )
                    
                    cat_new = cat.apply_mask_select(cat_new,self.mask_select_dict[mask_name_select],self.pix,tile_type=self.params["tile_type"],wcs=self.wcs)
                self.results[(mask_name, mask_name_select)] = cat_new
        
        self.results["empty"] = cat.cluster_catalogue()
        
        return 0

    #Apply MMF for input catalogue

    def extract_at_true_values_interp(self, true_catalogue, q_tensor, y_tensor,
                                       theta_vec, pix, wcs):
        """
        Fast extraction using 3D interpolation on q_tensor from find_clusters.

        Instead of recomputing MMF filters for each cluster (337 × MMF operations),
        interpolate SNR values directly from pre-computed q_tensor (0 × MMF operations).
        Speedup: ~4.4x overall (437→100 MMF evaluations with 100-point theta sampling).

        Parameters:
            true_catalogue: Input cluster catalogue with (ra, dec, theta_500)
            q_tensor: SNR tensor from find_clusters (nx, ny, n_theta)
            y_tensor: Flux tensor from find_clusters (nx, ny, n_theta)
            theta_vec: Array of theta_500 sampling points used in find_clusters
            pix: Pixelization object
            wcs: WCS projection (for CAR tiles)
        """
        import time
        from scipy.ndimage import map_coordinates, spline_filter
        from pixell import enmap

        t_start = time.time()
        n_clus = len(true_catalogue.catalogue["ra"])
        q_opt = np.zeros(n_clus)
        y0_est = np.zeros(n_clus)

        if self.params["tile_type"] != "car":
            raise NotImplementedError("extract_at_true_values_interp only supports CAR tile_type")

        # Convert (ra, dec) to pixel coordinates
        ra = true_catalogue.catalogue["ra"]
        dec = true_catalogue.catalogue["dec"]
        theta_true = true_catalogue.catalogue["theta_500"]

        pix_coords = enmap.sky2pix(
            (pix.nx, pix.ny), wcs,
            np.deg2rad([dec, ra]),
            corner=True
        )
        i_pix = pix_coords[0]
        j_pix = pix_coords[1]

        # 3D interpolation for each cluster (log-space theta interpolation)
        theta_idx_arr = np.arange(len(theta_vec))
        log_theta_vec = np.log(theta_vec)
        q_tensor_f = spline_filter(q_tensor, order=3, mode='nearest')
        y_tensor_f = spline_filter(y_tensor, order=3, mode='nearest')
        print("Interpolate input clusters:")
        for k in tqdm(range(n_clus)):
            # Fractional theta index in log space → trilinear interpolation in (x, y, log_theta)
            theta_float = np.interp(np.log(theta_true[k]), log_theta_vec, theta_idx_arr)

            # 3D interpolation: (i, j, theta_float)
            coords = np.array([[i_pix[k]], [j_pix[k]], [theta_float]])

            q_opt[k] = map_coordinates(q_tensor_f, coords, order=3, mode='nearest', prefilter=False)[0]
            y0_est[k] = map_coordinates(y_tensor_f, coords, order=3, mode='nearest', prefilter=False)[0]

        # Build result catalogue
        catalogue_interp = cat.cluster_catalogue()
        catalogue_interp.catalogue["q_opt"] = q_opt
        catalogue_interp.catalogue["y0"] = y0_est
        catalogue_interp.catalogue["theta_500"] = theta_true

        # Copy other fields
        for key in true_catalogue.catalogue.keys():
            if key not in ['q_opt', 'y0', 'theta_500']:
                catalogue_interp.catalogue[key] = true_catalogue.catalogue[key]

        # true_catalogue (per-tile input mock) has no pixel_ids of its own;
        # tag with this tile's field_id, matching extract_at_true_values (line ~1215)
        # and find_clusters (line ~951).
        catalogue_interp.catalogue["pixel_ids"] = [self.field_id] * n_clus

        self.catalogue_true_values = catalogue_interp

        t_end = time.time()
        if self.rank == 0:
            print(f"extract_at_true_values_interp: {n_clus} clusters in {t_end - t_start:.2f}s ({n_clus/(t_end - t_start):.1f} clusters/s)")
            print("extract_interp_q_opt:", q_opt)
            

        return catalogue_interp

    def extract_at_true_values(self,t_true,true_catalogue):

        if self.params["apod_type"] == "old":

            t_true = maps.multiply_t(self.mask_map,t_true)
            t_true = maps.filter_tmap(t_true,self.pix,self.params["lrange"],indices_filter=self.indices_filter)

        if self.params["tile_type"] == "healpix":

            n_clus = len(true_catalogue.catalogue["theta_x"])
            theta_0 = true_catalogue.catalogue["theta_x"]
            theta_1 = true_catalogue.catalogue["theta_y"]

        elif self.params["tile_type"] == "car":

            n_clus = len(true_catalogue.catalogue["ra"])
            theta_0 = true_catalogue.catalogue["ra"]
            theta_1 = true_catalogue.catalogue["dec"]

        q_opt = np.zeros((n_clus,len(self.params["comp_to_calculate"])))
        y0_est = np.zeros((n_clus,len(self.params["comp_to_calculate"])))
        q_opt_intp = np.zeros((n_clus,len(self.params["comp_to_calculate"])))
        y0_est_intp = np.zeros((n_clus,len(self.params["comp_to_calculate"])))
        q_opt_subpix = np.zeros((n_clus,len(self.params["comp_to_calculate"])))
        y0_est_subpix = np.zeros((n_clus,len(self.params["comp_to_calculate"])))

        if self.rank == 0:

            print("Extracting at true parameters")

        

        for i in tqdm(range(0, n_clus)):

            # z = true_catalogue.catalogue["redshift"][i]
            # M_500 = true_catalogue.catalogue["M500c"][i]/10
            z = 0.2 #????
            if "input_catalogue_theta" in self.params:  # fix filter size in FP mode
                M_500 = model.get_m_500(self.params["input_catalogue_theta"],z,self.cosmology) # 1e15 M_sun
            else:
                M_500 = model.get_m_500(true_catalogue.catalogue["theta_500"][i],z,self.cosmology) # 1e15 M_sun

            # if true_catalogue.catalogue["m_500"][i] < 5:continue
            
        

            if self.params["rSZ"] == True:

                tsz_sed_model = sed.tsz_model(T_e=true_catalogue.catalogue["T"][i])

                if self.params["integrate_bandpass"] == True:

                    tsz_sed = tsz_sed_model.get_sed_exp_bandpass(self.exp)

                elif self.params["integrate_bandpass"] == False:

                    tsz_sed = tsz_sed_model.get_sed(self.exp.nu_eff)

                tsz_sed = tsz_sed[self.params["freqs"]]


                a_matrix = self.params["a_matrix"][self.params["freqs"],:]
                a_matrix[:,0] = tsz_sed

            q_extracted,y0_extracted,q_extracted_intp,y0_extracted_intp, \
                q_extracted_subpix, y0_extracted_subpix, q_map = extract_at_input_value(t_true,
            self.inv_cov,
            self.pix,
            self.params["beam"],
            M_500,
            z,
            self.cosmology,
            self.params["norm_type"],
            theta_0[i],
            theta_1[i],
            self.params["lrange"],
            apod_type=self.params["apod_type"],
            f_sky_apo = self.params["f_sky_apo"],
            use_subpix_mmf=self.params.get("use_subpix_mmf", False),
            subpix_shape=self.params.get('subpix_shape', (2, 2)),
            mmf_type=self.params["mmf_type"],
            cmmf_type=self.params["cmmf_type"],
            cmmf_prec=self.cmmf,
            freqs=self.params["freqs"],
            exp=self.exp,
            comp_to_calculate=self.params["comp_to_calculate"],
            profile_type=self.params_model["profile_type"],
            indices_filter=self.indices_filter,
            tile_type=self.params["tile_type"],
            wcs=self.wcs,
            weights=self.weights
            )

            q_opt[i,:] = q_extracted
            y0_est[i,:] = y0_extracted
            q_opt_intp[i,:] = q_extracted_intp
            y0_est_intp[i,:] = y0_extracted_intp
            q_opt_subpix[i,:] = q_extracted_subpix
            y0_est_subpix[i,:] = y0_extracted_subpix
            # print("theta_500",true_catalogue.catalogue["theta_500"][i],"SNR",q_extracted, "y0", y0_extracted)
            # print("ACT", true_catalogue.catalogue["fixed_SNR"][i], true_catalogue.catalogue["fixed_y_c"][i])
            


        catalogue_at_true_values = cat.cluster_catalogue()

        catalogue_at_true_values.catalogue["q_opt"] = q_opt[:,0]
        catalogue_at_true_values.catalogue["y0"] = y0_est[:,0]
        catalogue_at_true_values.catalogue["q_opt_intp"] = q_opt_intp[:,0]
        catalogue_at_true_values.catalogue["y0_intp"] = y0_est_intp[:,0]
        catalogue_at_true_values.catalogue["q_opt_subpix"] = q_opt_subpix[:,0]
        catalogue_at_true_values.catalogue["y0_subpix"] = y0_est_subpix[:,0]
        catalogue_at_true_values.catalogue["theta_500"] = true_catalogue.catalogue["theta_500"]
        catalogue_at_true_values.catalogue["pixel_ids"] =  [self.field_id]*n_clus
        catalogue_at_true_values.catalogue["m_500"] = true_catalogue.catalogue["m_500"]
        catalogue_at_true_values.catalogue["z"] = true_catalogue.catalogue["z"]

        if self.params["tile_type"] == "healpix":

            catalogue_at_true_values.catalogue["theta_x"] = true_catalogue.catalogue["theta_x"]
            catalogue_at_true_values.catalogue["theta_y"] = true_catalogue.catalogue["theta_y"]

        elif self.params["tile_type"] == "car":

            catalogue_at_true_values.catalogue["dec"] = true_catalogue.catalogue["dec"]
            catalogue_at_true_values.catalogue["ra"] = true_catalogue.catalogue["ra"]

        for key, value in true_catalogue.catalogue.items():

            catalogue_at_true_values.catalogue["input_" + key] = np.copy(value)

        
        #Extraction of other components

        for i in range(1,len(self.params["comp_to_calculate"])):

            catalogue_at_true_values.catalogue["q_c" + str(i)] = q_opt[:,i]
            catalogue_at_true_values.catalogue["c" + str(i)] = y0_est[:,i]

        self.catalogue_true_values = catalogue_at_true_values

    def save_detections(self,q_opt,y0_est,maxs_idx,mask_name,j_theta, refined):

        thetas = np.ones_like(q_opt)*self.theta_500_vec[j_theta]
        ans = np.vstack([q_opt,y0_est,thetas,maxs_idx.T,
                        refined["i_interp"], refined["j_interp"], refined["q_interp"], refined["y0_interp"],
                        refined["i_com"], refined["j_com"], refined["q_com"], refined["y0_com"]])
        self.peak_info[mask_name] = np.hstack([self.peak_info[mask_name], ans])

#Apply MMF for input catalogue
def extract_at_input_value(t_true,inv_cov,pix,beam,M_500,z,cosmology,norm_type,
theta_0,theta_1,lrange,apod_type=None,f_sky_apo=1,mmf_type=None,cmmf_prec=None,cmmf_type=None,
freqs=None,exp=None,comp_to_calculate=None,profile_type=None,indices_filter=None,
tile_type="healpix",wcs=None,weights=None,use_subpix_mmf=False,subpix_shape=(2, 2)):
    print("extract_at_input_value",M_500,z)
    nfw = model.gnfw(M_500,z,cosmology,type=profile_type)

    q_opt = np.zeros(len(comp_to_calculate))
    y0_est = np.zeros(len(comp_to_calculate))
    q_opt_intp = np.zeros(len(comp_to_calculate))
    y0_est_intp = np.zeros(len(comp_to_calculate))
    q_opt_subpix = np.zeros(len(comp_to_calculate))
    y0_est_subpix = np.zeros(len(comp_to_calculate))


    if mmf_type == "standard" or (mmf_type == "spectrally_constrained" and cmmf_type == "one_dep"):

        tem = nfw.get_t_map_convolved(pix,
        exp,
        beam=beam,
        get_nc=False,
        sed=False,
        tile_type=tile_type,
        wcs=wcs
        )
        t_tem_norm = None

    elif mmf_type == "spectrally_constrained" and cmmf_type == "general":

        t_tem_norm,tem = nfw.get_t_map_convolved(pix,
        exp,
        beam=beam,
        get_nc=True,
        sed=False,
        tile_type=tile_type,
        wcs=wcs
        )

        t_tem_norm = t_tem_norm/nfw.get_y_norm(norm_type)
        t_tem_norm = maps.filter_tmap(t_tem_norm,pix,lrange)
    tem = tem/nfw.get_y_norm(norm_type)
    tem = maps.select_freqs(tem,freqs)
    tem = maps.filter_tmap(tem,pix,lrange,indices_filter=indices_filter)
    

    for k in range(0,len(comp_to_calculate)):

        if use_subpix_mmf:
            print("Using subpixel MMF")

            # subpixel shifts for interpolation, add on 2026-04-23
            subpix_result = get_mmf_q_map_subpix(
            t_true,
            tem,
            inv_cov,
            pix,
            mmf_type=mmf_type,
            cmmf_prec=cmmf_prec,
            tem_norm=t_tem_norm,
            comp=comp_to_calculate[k],
            f_sky_apo=f_sky_apo,
            subpix_shape=subpix_shape
            )

            q_map_tem = subpix_result["q_maps"][0, 0]
            y_map = subpix_result["y_maps"][0, 0]
            std = subpix_result["std"]

        else:
            print("Not using subpixel MMF")

            subpix_result = None
            q_map_tem,y_map,std = get_mmf_q_map(t_true,tem,inv_cov,pix,mmf_type=mmf_type,
            cmmf_prec=cmmf_prec,tem_norm=t_tem_norm,comp=comp_to_calculate[k],
                                               f_sky_apo = f_sky_apo)
        
        

        if weights is not None:
                
            q_map_tem = q_map_tem*weights

        if k == 0:

            q_map = q_map_tem

        if tile_type == "healpix":

            j_clus = int(np.floor(theta_0/pix.dx))
            i_clus = int(pix.ny-np.ceil(theta_1/pix.dy))

            q_opt[k] = q_map_tem[i_clus,j_clus]
            y0_est[k] = y_map[i_clus,j_clus]
            q_opt_subpix[k] = q_opt[k]
            y0_est_subpix[k] = y0_est[k]

        elif tile_type == "car":

            ra = theta_0
            dec = theta_1

            from pixell import enmap
            
            pix_coords = enmap.sky2pix((pix.nx,pix.ny),wcs,np.deg2rad([dec,ra]), corner = True)
            
            i_clus,j_clus = np.round(pix_coords).astype(int)
            q_opt[k] = q_map_tem[i_clus,j_clus]
            y0_est[k] = y_map[i_clus,j_clus]
            q_opt_subpix[k] = q_opt[k]
            y0_est_subpix[k] = y0_est[k]
            
            i_clus,j_clus = pix_coords
            
            from scipy.ndimage import map_coordinates
            q_opt_intp[k] = map_coordinates(q_map_tem,
                                        [[i_clus], [j_clus]],
                                        order=2,        
                                        mode="nearest"  
                                    )[0]
            y0_est_intp[k] = map_coordinates(y_map,
                                        [[i_clus], [j_clus]],
                                        order=2,        
                                        mode="nearest"  
                                    )[0]
            if use_subpix_mmf:

                q_opt_subpix[k], y0_est_subpix[k], si, sj, i_coarse, j_coarse, \
                i_pix_sampled, j_pix_sampled = sample_subpix_result_at_pixcoord(
                    subpix_result, pix_coords)

            
            
            
    
    return q_opt,y0_est,q_opt_intp,y0_est_intp,q_opt_subpix, y0_est_subpix, q_map


# interpolation in blind mode, add on 2026-04-23
def refine_detection_positions_2d(q_map, y_map, i_peak, j_peak, q_th, mask_peak_finding, params):
    i_peak = np.asarray(i_peak, dtype=np.int64)
    j_peak = np.asarray(j_peak, dtype=np.int64)
    n_det = len(i_peak)

    out = {
        "i_interp": i_peak.astype(float),
        "j_interp": j_peak.astype(float),
        "q_interp": q_map[i_peak, j_peak] if n_det > 0 else np.empty(0),
        "y0_interp": y_map[i_peak, j_peak] if n_det > 0 else np.empty(0),
        "i_com": i_peak.astype(float),
        "j_com": j_peak.astype(float),
        "q_com": q_map[i_peak, j_peak] if n_det > 0 else np.empty(0),
        "y0_com": y_map[i_peak, j_peak] if n_det > 0 else np.empty(0),
    }

    if n_det == 0:
        return out

    radius = params.get("blind_position_interp_radius_pix", 1.0)
    oversample = params.get("blind_position_interp_oversample", 10)
    order = params.get("blind_position_interp_order", 2)
    offsets = np.linspace(-radius, radius, int(2 * radius * oversample) + 1)
    # print("interpolation: radiue = ", radius, ",  oversample = ", oversample, ",  order = ", order)  # Debug only

    mask_float = mask_peak_finding.astype(float)
    
    for k, (i0, j0) in enumerate(zip(i_peak, j_peak)):
        ii, jj = np.meshgrid(i0 + offsets, j0 + offsets, indexing="ij")
        coords = np.vstack([ii.ravel(), jj.ravel()])

        valid = ndimage.map_coordinates(mask_float, coords, order=0, mode="constant", cval=0.) > 0.5
        q_vals = ndimage.map_coordinates(q_map, coords, order=order, mode="nearest")
        q_vals[~valid] = -np.inf

        if np.any(np.isfinite(q_vals)):
            ind = np.argmax(q_vals)
            out["i_interp"][k] = coords[0, ind]
            out["j_interp"][k] = coords[1, ind]
            out["q_interp"][k] = q_vals[ind]
            out["y0_interp"][k] = ndimage.map_coordinates(
                y_map, [[out["i_interp"][k]], [out["j_interp"][k]]],
                order=order, mode="nearest"
            )[0]

    # Center of mass calculation disabled (not needed)
    # labels, _ = ndimage.label((q_map > q_th) & (mask_peak_finding != 0))
    # for k, (i0, j0) in enumerate(zip(i_peak, j_peak)):
    #     label_id = labels[i0, j0]
    #     if label_id == 0:
    #         continue
    #
    #     i_com, j_com = ndimage.center_of_mass(q_map, labels=labels, index=label_id)
    #     if not np.isfinite(i_com) or not np.isfinite(j_com):
    #         continue
    #
    #     out["i_com"][k] = i_com
    #     out["j_com"][k] = j_com
    #     out["q_com"][k] = ndimage.map_coordinates(q_map, [[i_com], [j_com]], order=order, mode="nearest")[0]
    #     out["y0_com"][k] = ndimage.map_coordinates(y_map, [[i_com], [j_com]], order=order, mode="nearest")[0]

    return out


def apply_mask_peak_finding(tensor,mask):

    for i in range(0,tensor.shape[2]):

        tensor[:,:,i] *= mask

    return tensor

#Engine for cluster finding

def make_detections(q_tensor,q_th,pix,detection_method="maxima"):

    indices = np.where(q_tensor > q_th)

    coords = np.zeros((len(indices[0]),2),dtype=int)
    coords[:,0] = indices[0]
    coords[:,1] = indices[1]
    theta_coord = np.array(indices[2])

    if coords.shape[0] == 0:

        ret = ([],[],[])

    else:

        if detection_method == "DBSCAN":

            clust = DBSCAN(eps=2)
            clust.fit(coords)
            labels = clust.labels_
            n_clusters = np.max(labels)+1

            i_opt_vec = np.zeros(n_clusters,dtype=int)
            j_opt_vec = np.zeros(n_clusters,dtype=int)
            theta_opt_vec = np.zeros(n_clusters,dtype=int)

            for i in range(0,n_clusters):

                indices = np.where(labels==i)
                q_tensor_select = q_tensor[coords[:,0][indices],coords[:,1][indices],theta_coord[indices]]
                opt_index = np.argmax(q_tensor_select)
                i_opt_vec[i]  = coords[:,0][indices][opt_index]
                j_opt_vec[i] = coords[:,1][indices][opt_index]
                theta_opt_vec[i] = theta_coord[indices][opt_index]
                q_opt = q_tensor_select[opt_index]

            ret = (i_opt_vec,j_opt_vec,theta_opt_vec)

        elif detection_method in ["maxima", "maxima_lomem"]:

            q_tensor_maxima = q_tensor[:,:,:].copy()
            q_tensor_maxima[np.where(q_tensor_maxima < q_th)] = 0.

            maxs_idx = get_maxima(get_maxima_mask(q_tensor_maxima))

            i = np.array(maxs_idx[:,0]).astype(int)
            j = np.array(maxs_idx[:,1]).astype(int)
            theta = np.array(maxs_idx[:,2]).astype(int)

            ret = (i,j,theta)

    return ret

def make_detections2(qmap,ymap,mask_peak_finding,q_th,zmax, params=None):

    coords = np.where(np.logical_and(mask_peak_finding, (qmap > q_th)))  # Assume binary mask
    maxs_idx = get_maxima(get_maxima_mask_2d(qmap, coords) & zmax)
    indices = tuple(maxs_idx.astype(int).T) # (idx[:,0], idx[;,1])
    q_opt = qmap[indices]
    y0_est = ymap[indices]

    refined = refine_detection_positions_2d(
    qmap, ymap, maxs_idx[:,0], maxs_idx[:,1],
    q_th, mask_peak_finding, params or {}
    )

    return q_opt,y0_est,maxs_idx, refined ## arrays: (Ndetections,), (Ndetections,), (Ndetections, 2)

#Mask for peak finding

def get_maxima_mask(r_input):

    # Along each axis a voxel is a local maximum if it is strictly above its next
    # neighbour and either strictly above the previous one, or exactly equal to the
    # previous one while being strictly above the one beyond that.  The second clause
    # lets a two-element plateau count exactly once (at its far end) and only when the
    # plateau really is a maximum, not a flat step on a slope.
    #
    # The previous implementation used argmax([next, prev, centre], axis=0) == 2, which
    # demanded strict inequality on BOTH sides.  Since argmax returns the FIRST maximum,
    # an exact tie between adjacent layers rejected both members of the tied pair and the
    # detection disappeared from the catalogue altogether.  Such ties are not pathological:
    # the true maximum of q(theta) generally falls between two grid layers, so those two
    # layers are nearly equal, and once their difference drops below one float32 ulp they
    # become bit-identical.  See the note at the maxima_lomem theta test in find_clusters
    # for measured rates and a worked example.

    r = np.full((r_input.shape[0]+4, r_input.shape[1]+4, r_input.shape[2]+4),
                -np.inf, dtype=r_input.dtype)
    r[2:-2,2:-2,2:-2] = r_input
    c = r[2:-2,2:-2,2:-2]

    def axis_max(nxt, prv, prv2):

        return (c > nxt) & ((c > prv) | ((c == prv) & (c > prv2)))

    xmax = axis_max(r[3:-1,2:-2,2:-2], r[1:-3,2:-2,2:-2], r[:-4,2:-2,2:-2])
    ymax = axis_max(r[2:-2,3:-1,2:-2], r[2:-2,1:-3,2:-2], r[2:-2,:-4,2:-2])
    zmax = axis_max(r[2:-2,2:-2,3:-1], r[2:-2,2:-2,1:-3], r[2:-2,2:-2,:-4])

    return xmax&ymax&zmax

def get_maxima_mask_2d(r_input, coords=None):

    r = np.ones((r_input.shape[0]+2, r_input.shape[1]+2)) * -np.inf
    if coords is None:
        r[1:-1, 1:-1] = r_input
    else:
        r[1:-1, 1:-1] = 0
        r[1:-1, 1:-1][coords] = r_input[coords]

    xmax = np.argmax([r[2:, 1:-1], r[:-2, 1:-1], r[1:-1, 1:-1]], axis=0) == 2
    ymax = np.argmax([r[1:-1, 2:], r[1:-1, :-2], r[1:-1, 1:-1]], axis=0) == 2

    return xmax&ymax

#Peak finding

def get_maxima(max_mask):

    maxima = np.argwhere(max_mask == True)

    return maxima

#Get SNR map

def get_mmf_q_map(tmap,tem,inv_cov,pix,
mmf_type="standard",cmmf_prec=None,comp=0,tem_norm=None, 
                 f_sky_apo = 1, field_id = 0,  noise_type="fid",
                 path_save=""):
    
    n_freqs = tmap.shape[2]
    y_map = np.zeros((pix.nx,pix.ny), dtype=tmap.dtype)
    
    tem_fft = maps.get_fft_f(tem,pix)
    tem_save = tem.copy() # debug

        
    filter_fft = get_tem_conv_fft(pix,tem_fft, inv_cov,mmf_type=mmf_type,
    cmmf_prec=cmmf_prec,comp=comp, field_id=field_id)
    
    filter = np.asarray(maps.get_ifft_f(filter_fft,pix).real,dtype=tmap.dtype)

    

    tem[:] = maps.get_tmap_times_fvec(tem,cmmf_prec.a_matrix[:,comp]).real #new line

    if tem_norm is None:

        tem_norm = tem

    else:

        tem_norm = maps.get_tmap_times_fvec(tem_norm,cmmf_prec.a_matrix[:,comp]).real #new line

        del tem

    # debug save
    if path_save:
        # np.savez(path_save, filter=filter, inv_cov=inv_cov, tem=tem, filter_fft=filter_fft, tem_fft=tem_fft, pix=pix, tmap = tmap)
        np.savez(path_save, filter=filter, tem=tem, pix=pix, filter_fft=filter_fft, tem_fft = tem_fft)

    del filter_fft, tem_fft
    
    y_save = np.zeros(tmap.shape)
    for i in range(n_freqs):
        y = sg.fftconvolve(tmap[:,:,i],filter[:,:,i],mode="same")*pix.dx*pix.dy
        y_map += y
        y_save[:, :, i] = y

    y_map = np.asarray(y_map, dtype=tmap.dtype)

    del y

    norm = np.sum(tem_norm*filter)*pix.dx*pix.dy

    # del tem_norm, filter

    y_map = y_map/norm

    if noise_type=="fid":
        std = 1./np.sqrt(norm)
    elif noise_type=="cells":
        std = MakeNoiseMap_simple(mapData = y_map) 

    std = std/f_sky_apo
    
    q_map = y_map/std
   
    return q_map,y_map,std

# based on get_mmf_q_map, with subpixel shifts for interpolation, add on 2026-04-23
def get_mmf_q_map_subpix(tmap, tem, inv_cov, pix,
mmf_type="standard", cmmf_prec=None, comp=0, tem_norm=None,
                         f_sky_apo=1, field_id=0, noise_type="fid",
                         subpix_shape=(3, 3),
                         path_save=""):

    if noise_type != "fid":
        raise NotImplementedError(
            "get_mmf_q_map_subpix currently supports only noise_type='fid'"
        )
    print("subpix_shape", subpix_shape)
    n_sub_i, n_sub_j = subpix_shape
    n_freqs = tmap.shape[2]

    tem_local = tem.copy()
    if tem_norm is None:
        tem_norm_local = None
    else:
        tem_norm_local = tem_norm.copy()

    tem_fft = maps.get_fft_f(tem_local, pix)

    filter_fft = get_tem_conv_fft(
        pix, tem_fft, inv_cov,
        mmf_type=mmf_type,
        cmmf_prec=cmmf_prec,
        comp=comp,
        field_id=field_id
    )

    filter0 = np.asarray(maps.get_ifft_f(filter_fft, pix).real, dtype=tmap.dtype)

    tem_local[:] = maps.get_tmap_times_fvec(
        tem_local, cmmf_prec.a_matrix[:, comp]
    ).real

    if tem_norm_local is None:
        tem_norm_eff = tem_local
    else:
        tem_norm_eff = maps.get_tmap_times_fvec(
            tem_norm_local, cmmf_prec.a_matrix[:, comp]
        ).real

    norm = np.sum(tem_norm_eff * filter0) * pix.dx * pix.dy
    std = 1.0 / np.sqrt(norm)
    std = std / f_sky_apo

    lx, ly = maps.rmap(pix).get_lxly()

    shift_i = np.arange(n_sub_i) * pix.dx / n_sub_i
    shift_j = np.arange(n_sub_j) * pix.dy / n_sub_j

    q_maps = np.zeros((n_sub_i, n_sub_j, pix.nx, pix.ny), dtype=tmap.dtype)
    y_maps = np.zeros((n_sub_i, n_sub_j, pix.nx, pix.ny), dtype=tmap.dtype)

    for si in range(n_sub_i):
        for sj in range(n_sub_j):

            phase = np.exp(
                1j * (lx * shift_i[si] + ly * shift_j[sj])
            )[:, :, None]

            filter_fft_shift = filter_fft * phase
            filter_shift = np.asarray(
                maps.get_ifft_f(filter_fft_shift, pix).real,
                dtype=tmap.dtype
            )

            y_map_shift = np.zeros((pix.nx, pix.ny), dtype=tmap.dtype)

            for i_freq in range(n_freqs):
                y = sg.fftconvolve(
                    tmap[:, :, i_freq],
                    filter_shift[:, :, i_freq],
                    mode="same"
                ) * pix.dx * pix.dy
                y_map_shift += y

            y_map_shift = np.asarray(y_map_shift, dtype=tmap.dtype)
            y_map_shift = y_map_shift / norm
            q_map_shift = y_map_shift / std

            y_maps[si, sj, :, :] = y_map_shift
            q_maps[si, sj, :, :] = q_map_shift

    subpix_result = {
        "q_maps": q_maps,
        "y_maps": y_maps,
        "std": std,
        "subpix_shape": (n_sub_i, n_sub_j),
        "shift_i": shift_i,
        "shift_j": shift_j,
    }

    return subpix_result
def sample_subpix_result_at_pixcoord(subpix_result, pix_coords):

    n_sub_i, n_sub_j = subpix_result["subpix_shape"]
    q_maps = subpix_result["q_maps"]
    y_maps = subpix_result["y_maps"]

    i_pix = float(pix_coords[0])
    j_pix = float(pix_coords[1])

    i_sub = int(np.round(n_sub_i * i_pix))
    j_sub = int(np.round(n_sub_j * j_pix))
    
    si = i_sub % n_sub_i
    sj = j_sub % n_sub_j

    i_coarse = i_sub // n_sub_i
    j_coarse = j_sub // n_sub_j

    nx = q_maps.shape[2]
    ny = q_maps.shape[3]

    i_coarse = np.clip(i_coarse, 0, nx - 1)
    j_coarse = np.clip(j_coarse, 0, ny - 1)

    q_val = q_maps[si, sj, i_coarse, j_coarse]
    y_val = y_maps[si, sj, i_coarse, j_coarse]

    i_pix_sampled = i_sub / float(n_sub_i)
    j_pix_sampled = j_sub / float(n_sub_j)

    return q_val, y_val, si, sj, i_coarse, j_coarse, i_pix_sampled, j_pix_sampled



def MakeNoiseMap_simple(mapData, noiseGridArcmin = 80, numBins=1):
    
    getPixelSizeDeg = 0.5/60
    gridSize=int(round(noiseGridArcmin/60./getPixelSizeDeg))
    overlapPix= int(gridSize/2)
    numXChunks=mapData.shape[1]/gridSize
    numYChunks=mapData.shape[0]/gridSize
    print(gridSize, overlapPix, numXChunks, numYChunks)
    yChunks=np.linspace(0, mapData.shape[0], int(numYChunks+1), dtype = int)
    xChunks=np.linspace(0, mapData.shape[1], int(numXChunks+1), dtype = int)

    RMSMap=np.zeros(mapData.shape, dtype = np.float32)

    for i in range(len(yChunks)-1):
        for k in range(len(xChunks)-1):
            
            y0=yChunks[i]-overlapPix
            y1=yChunks[i+1]+overlapPix
            x0=xChunks[k]-overlapPix
            x1=xChunks[k+1]+overlapPix
            if y0 < 0:
                y0=0
            if y1 > mapData.shape[0]:
                y1=mapData.shape[0]
            if x0 < 0:
                x0=0
            if x1 > mapData.shape[1]:
                x1=mapData.shape[1]
            chunkValues=mapData[y0:y1, x0:x1]

            chunkMean=np.mean(chunkValues)
            chunkRMS=np.std(chunkValues)
            sigmaClip=3.0
            for c in range(10):
                mask=np.less(abs(chunkValues), abs(chunkMean+sigmaClip*chunkRMS))
                chunkMean=np.mean(chunkValues[mask])
                chunkRMS=np.std(chunkValues[mask])
            RMSMap[y0:y1, x0:x1]=chunkRMS
    return RMSMap


    

def get_tem_conv_fft(pix_tem,tem_fft,inv_cov,mmf_type="standard",cmmf_prec=None,comp=0, field_id = 0):

    tem_fft = maps.reshape_ell_matrix(tem_fft, inv_cov.shape[:2]) # Match to cut inv_cov

    if mmf_type == "standard":    

        tem_fft = maps.get_tmap_times_fvec(tem_fft,cmmf_prec.a_matrix[:,comp]) #new line
        filter_fft = utils.get_inv_cov_conjugate(tem_fft,inv_cov)

        # np.savez( "/shared_home/Ziyang/Cluster_Finder/realisation_1/test_"+str(field_id)+".npz", filter = filter_fft, tem = tem_fft)
        # print("save", "/shared_home/Ziyang/Cluster_Finder/realisation_1/test_"+str(field_id)+".npz");exit()

    elif mmf_type == "spectrally_constrained": #doesn't admit changing component order

        if cmmf_prec.cmmf_type == "one_dep":

            n_freq = tem_fft.shape[2]
            a_dot_b =  maps.get_tmap_from_map(cmmf_prec.a_dot_b,n_freq)
            b_dot_b = maps.get_tmap_from_map(cmmf_prec.b_dot_b,n_freq)
            b = cmmf_prec.a_matrix[:,1] #cmmf_prec.sed_b
            a = cmmf_prec.a_matrix[:,0] #cmmf_prec.sed_a

            tem_fft_b = maps.get_tmap_times_fvec(tem_fft,b)
            tem_fft = maps.get_tmap_times_fvec(tem_fft,a)

            filter_fft = utils.get_inv_cov_conjugate(tem_fft,inv_cov) - a_dot_b/b_dot_b*utils.get_inv_cov_conjugate(tem_fft_b,inv_cov)

        elif cmmf_prec.cmmf_type == "general":

            filter_fft_0 = cmmf_prec.w_norm_list[comp]
            filter_fft = filter_fft_0*tem_fft

    filter_fft[np.isnan(filter_fft)] = 0.
    filter_fft = maps.reshape_ell_matrix(filter_fft, (pix_tem.nx, pix_tem.ny)) # Expand back to expected shape

    return filter_fft

#Class to compute weights for spectrally constrained MMF

class scmmf_precomputation:

    #sed's in muK

    def __init__(self,pix=None,freqs=None,inv_cov=None,lrange=[0,1000000],beam_type="gaussian",exp=None,
    mask=None,cmmf_type="one_dep",a_matrix=None,n_dep=1,comp_to_calculate=[0],mmf_type="standard"):

        self.cmmf_type = cmmf_type
        self.a_matrix = a_matrix

    
        if mask is None:

            mask = np.ones((pix.nx,pix.ny,len(freqs)))

        if self.cmmf_type == "one_dep" and mmf_type == "spectrally_constrained":

            self.sed_a = a_matrix[:,0][freqs]
            self.sed_b = a_matrix[:,1][freqs]

            a_map_fft = np.ones((pix.nx,pix.nx,len(freqs)))
            a_map_fft = maps.get_tmap_times_fvec(a_map_fft,self.sed_a)

            a_map_fft = maps.get_map_convolved_fft(a_map_fft,pix,freqs,beam_type,mask,lrange,exp)

            # TODO: More efficient to do next line earlier, but requires modifications to the functions in maps

            a_map_fft = maps.reshape_ell_matrix(a_map_fft, inv_cov.shape[:2]) # Cut to correct lrange

            b_map_fft = np.ones((pix.nx,pix.nx,len(freqs)))
            b_map_fft = maps.get_tmap_times_fvec(b_map_fft,self.sed_b)

            b_map_fft = maps.get_map_convolved_fft(b_map_fft,pix,freqs,beam_type,mask,lrange,exp)
            b_map_fft = maps.reshape_ell_matrix(b_map_fft, inv_cov.shape[:2])

            self.a_dot_b = utils.get_inv_cov_dot(np.conjugate(a_map_fft),inv_cov,b_map_fft)
            self.b_dot_b = utils.get_inv_cov_dot(np.conjugate(b_map_fft),inv_cov,b_map_fft)

            del a_map_fft
            del b_map_fft

        elif self.cmmf_type == "general" and mmf_type == "spectrally_constrained":

            #a_matrix is f x c matrix (f number of frequencies, c number of components, 1st SZ, rest to deproject)

            a_matrix_fft = np.zeros((pix.nx,pix.nx,len(freqs),self.a_matrix.shape[1]))

            for i in range(0,a_matrix.shape[1]):

                a_matrix_fft_i = np.ones((pix.nx,pix.nx,len(freqs)))
                a_matrix_fft_i = maps.get_tmap_times_fvec(a_matrix_fft_i,self.a_matrix[freqs,i])
                a_matrix_fft[:,:,:,i] = maps.get_map_convolved_fft(a_matrix_fft_i,pix,freqs,beam_type,mask,lrange,exp)

            del a_matrix_fft_i

            a_matrix_fft = maps.reshape_ell_matrix(a_matrix_fft, inv_cov.shape[:2])

            pre = np.einsum("abcd,abde->abce",inv_cov,a_matrix_fft)
            dot = utils.invert_cov(np.einsum("abdc,abde->abce",a_matrix_fft,pre))

            del a_matrix_fft

            W = np.einsum("abcd,abed->abce",dot,pre)
            cov = utils.invert_cov(inv_cov)

            self.w_norm_list = []

            for i in range(0,len(comp_to_calculate)):

                w = W[:,:,comp_to_calculate[i]]

                sigma_y = utils.get_inv_cov_dot(np.conjugate(w),cov,w)

                w_norm = np.zeros(w.shape)

                for i in range(0,len(freqs)):

                    w_norm[:,:,i] = w[:,:,i]/sigma_y

                self.w_norm_list.append(w_norm)

        elif mmf_type == "standard":

            r = 1.

        else:

            if mask is None:

                mask = np.ones((pix.nx,pix.ny,len(freqs)))

            if self.cmmf_type == "one_dep" and mmf_type == "spectrally_constrained":

                self.sed_a = a_matrix[:,0][freqs]
                self.sed_b = a_matrix[:,1][freqs]

                a_map_fft = np.ones((pix.nx,pix.nx,len(freqs)))
                a_map_fft = maps.get_tmap_times_fvec(a_map_fft,self.sed_a)

                a_map_fft = maps.get_map_convolved_fft(a_map_fft,pix,freqs,beam_type,mask,lrange,exp)

                b_map_fft = np.ones((pix.nx,pix.nx,len(freqs)))
                b_map_fft = maps.get_tmap_times_fvec(b_map_fft,self.sed_b)

                b_map_fft = maps.get_map_convolved_fft(b_map_fft,pix,freqs,beam_type,mask,lrange,exp)

                self.a_dot_b = utils.get_inv_cov_dot(np.conjugate(a_map_fft),inv_cov,b_map_fft)
                self.b_dot_b = utils.get_inv_cov_dot(np.conjugate(b_map_fft),inv_cov,b_map_fft)

            elif self.cmmf_type == "general" and mmf_type == "spectrally_constrained":

                #a_matrix is f x c matrix (f number of frequencies, c number of components, 1st SZ, rest to deproject)

                a_matrix_fft = np.zeros((pix.nx,pix.nx,len(freqs),self.a_matrix.shape[1]))

                for i in range(0,a_matrix.shape[1]):

                    a_matrix_fft_i = np.ones((pix.nx,pix.nx,len(freqs)))
                    a_matrix_fft_i = maps.get_tmap_times_fvec(a_matrix_fft_i,self.a_matrix[freqs,i])
                    a_matrix_fft[:,:,:,i] = maps.get_map_convolved_fft(a_matrix_fft_i,pix,freqs,beam_type,mask,lrange,exp)

                self.a_matrix_fft = a_matrix_fft

                pre = np.einsum("abcd,abde->abce",inv_cov,a_matrix_fft)
                dot = utils.invert_cov(np.einsum("abdc,abde->abce",a_matrix_fft,pre))
                self.W = np.einsum("abcd,abed->abce",dot,pre)
                cov = utils.invert_cov(inv_cov)

                self.w_list = []
                self.sigma_y_list = []
                self.w_norm_list = []

                for i in range(0,len(comp_to_calculate)):

                    e_map = np.zeros((pix.nx,pix.nx,self.a_matrix.shape[1]))
                    e_map[:,:,comp_to_calculate[i]] = 1.

                    w = np.einsum("ija,ijab->ijb",e_map,self.W)

                    sigma_y = utils.get_inv_cov_dot(np.conjugate(w),cov,w)

                    w_norm = np.zeros(w.shape)

                    for i in range(0,len(freqs)):

                        w_norm[:,:,i] = w[:,:,i]/sigma_y

                    self.w_list.append(w)
                    self.sigma_y_list.append(sigma_y)
                    self.w_norm_list.append(w_norm)
        
def get_a_matrix_cib(params_szifi,params_model,data_file):

    cib = sed.cib_model(params_model=params_model)
    cib_sed = cib.get_sed_muK_experiment(experiment=data_file["experiment"],bandpass=params_szifi["integrate_bandpass"])
    cib.get_sed_first_moments_experiment(experiment=data_file["experiment"],
    bandpass=params_szifi["integrate_bandpass"],moment_parameters=params_szifi["deproject_cib"])

    freqs = params_szifi["freqs"]

    a_matrix = np.zeros((len(freqs),len(params_szifi["deproject_cib"])))

    if params_szifi["deproject_cib"] == ["cib"]:

        a_matrix = np.zeros((len(freqs),2))
        a_matrix[:,0] = data_file["experiment"].tsz_sed[freqs]
        a_matrix[:,1] = cib_sed[freqs]

    elif params_szifi["deproject_cib"] == ["cib","betaT"]:

        a_matrix = np.zeros((len(freqs),3))
        a_matrix[:,0] = data_file["experiment"].tsz_sed[freqs]
        a_matrix[:,1] = cib_sed[freqs]
        a_matrix[:,2] = cib.moments["betaT"][freqs]

    elif params_szifi["deproject_cib"] == ["cib","beta"]:

        a_matrix = np.zeros((len(freqs),3))
        a_matrix[:,0] = data_file["experiment"].tsz_sed[freqs]
        a_matrix[:,1] = cib_sed[freqs]
        a_matrix[:,2] = cib.moments["beta"][freqs]

    elif params_szifi["deproject_cib"] == ["cib","betaT","beta"]:

        a_matrix = np.zeros((len(freqs),3))
        a_matrix[:,0] = data_file["experiment"].tsz_sed[freqs]
        a_matrix[:,1] = cib_sed[freqs]
        a_matrix[:,2] = cib.moments["betaT"][freqs]
        a_matrix[:,3] = cib.moments["beta"][freqs]

    params_szifi["a_matrix"] = a_matrix
