from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
import warnings
import h5py
import numpy as np
import chaospy
import yaml
from tqdm.auto import tqdm, trange
import os
import logging
import sys


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)



def set_logger_level(logger, level):
    logging_levels = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }

    logger.setLevel(logging_levels[level])


def get_logger(filepath, logging_level=None):
    if logging_level is None:
        if "PYTHON_LOGGER_LEVEL" in os.environ:
            logging_level = os.environ["PYTHON_LOGGER_LEVEL"]
        else:
            logging_level = "info"

    logger = logging.getLogger(os.path.basename(filepath)[:10])

    if len(logger.handlers) == 0:
        log_formatter = logging.Formatter(
            fmt="%(asctime)s %(name)10s %(levelname).3s   %(message)s ",
            datefmt="%y-%m-%d %H:%M:%S",
            style="%",
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(log_formatter)
        logger.addHandler(stream_handler)
        logger.propagate = False
        set_logger_level(logger, logging_level)

    return logger

LOGGER = get_logger(__file__)

ERRVAL = -1e100

global mi
mi = None

DISTRIBUTIONS = ["flat", "uniform"]
SEQUENCES = [
    "random",
    "korobov",
    "latin_hypercube",
    "sobol",
    "halton",
    "hammersley",
]


def load_config(filename_config):

    with open(filename_config) as f:
        conf = yaml.load(f, Loader=yaml.Loader)
    if "pycosmo_param_file" in conf.keys():
        if conf["pycosmo_param_file"].lower() == "default":
            conf["pycosmo_param_file"] = None
    conf.setdefault("pycosmo_param_file", None)
    conf.setdefault("output_precision", np.float32)
    conf.setdefault("use_cross_correlations", True)

    return conf


def load_data(path):
    with h5py.File(path) as fh5:
        params_cosmo = np.array(fh5["cosmo_params"])
        cl = np.array(fh5["Cl"])
    cl = cl.reshape(cl.shape[0], -1, order="C")
    return params_cosmo, cl


def get_ranges(filename_config):
    conf = load_config(filename_config)
    ranges = {}
    for p in conf["variable_cosmology_parameters"]:
        ranges[p["name"]] = [p["min"], p["max"]]

    return ranges


def get_param_lims(variable_cosmology_parameters, cosmology_params_fixed={}):
    n_params = len(variable_cosmology_parameters)
    lims = np.zeros([n_params, 2])
    index = 0
    for i in range(n_params):
        param = variable_cosmology_parameters[i]["name"]
        if param not in cosmology_params_fixed.keys():
            lims[index, 0] = variable_cosmology_parameters[i]["min"]
            lims[index, 1] = variable_cosmology_parameters[i]["max"]

            try:
                lims[index, 0] = variable_cosmology_parameters[i]["priormin"]
                lims[index, 1] = variable_cosmology_parameters[i]["priormax"]
                LOGGER.info("setting prior from config for param: {}".format(param))
            except Exception:
                pass

            index += 1
    return lims[:index]


def get_chaospy_distribution(spec):
    spec.setdefault("distribution", "uniform")
    spec.setdefault("distribution_kwargs", {})

    d = None
    if spec["distribution"].lower() in ["flat", "uniform"]:
        d = chaospy.Uniform(lower=spec["min"], upper=spec["max"])

    elif spec["distribution"].lower() in ["gaussian", "normal"]:
        d = chaospy.TruncNormal(
            lower=spec["min"], upper=spec["max"], **spec["distribution_kwargs"]
        )
    else:
        raise Exception(
            "distribution {} not implemented, "
            "currently available {}".format(spec["distribution"], str(DISTRIBUTIONS))
        )

    return d

def get_chaospy_joint(list_distributions):
    return chaospy.J(*list_distributions)

def sample_sequence(d, n_samples, n_gridpoints=None, first=0, sequence="sobol"):
    if sequence not in SEQUENCES:
        raise Exception(
            "sequence {} not supported, use {}".format(sequence, str(SEQUENCES))
        )

    # decide whether to load full sequcence or continue it
    if n_gridpoints is None:
        n_gridpoints = n_samples + first

    samples = d.sample(size=n_gridpoints, rule=sequence).T

    return samples[first : first + n_samples, :]


def get_grid(conf_params, n_samples, n_gridpoints=None, first=0, sequence="sobol"):
    n_params = len(conf_params)
    list_param_names = [p["name"] for p in conf_params]

    list_distrib = [get_chaospy_distribution(conf_params[ip]) for ip in range(n_params)]
    prior = get_chaospy_joint(list_distrib)
    samp = sample_sequence(
        d=prior,
        sequence=sequence,
        n_gridpoints=n_gridpoints,
        n_samples=n_samples,
        first=first,
    )

    dtype = dict(formats=["f4"] * n_params, names=list_param_names)
    params_grid = np.zeros(len(samp), dtype=dtype)
    for i, par_name in enumerate(list_param_names):
        params_grid[par_name] = samp[:, i]
    return params_grid

def get_params_grid(
    conf, n_samples, n_gridpoints=None, first=0, sequence="sobol", nuisance_params=None
):
    params = conf["variable_cosmology_parameters"]

    if nuisance_params:
        params += conf["nuisance_params"]

    return get_grid(
        conf_params=params,
        n_samples=n_samples,
        n_gridpoints=n_gridpoints,
        first=first,
        sequence=sequence,
    )

def get_gaussian_priors(start_of_param_name, mean, sigma, params_cosmo):
    if not isinstance(start_of_param_name, (list, tuple, np.ndarray)):
        start_of_param_name = [start_of_param_name]
        mean = [mean]
        sigma = [sigma]
    g_prior = []
    g_mean = []
    g_sigma = []
    for i, s in enumerate(start_of_param_name):
        g_p, g_m, g_s = _get_gaussian_priors_for_oneparam(
            s, mean[i], sigma[i], params_cosmo
        )
        g_prior += g_p
        g_mean.append(g_m)
        g_sigma.append(g_s)
    return g_prior, np.concatenate(g_mean), np.concatenate(g_sigma)


def _get_gaussian_priors_for_oneparam(param, mean, sigma, params_cosmo):
    g_prior = []
    try:
        params_cosmo = params_cosmo.dtype.names
    except Exception:
        pass
    for i, p in enumerate(params_cosmo):
        if p.startswith(param):
            g_prior.append(i)
    n_par = len(g_prior)
    return g_prior, np.ones(n_par) * mean, np.ones(n_par) * sigma


def run_mcmc_with_emulator(
    n_samples,
    cl_obs,
    C,
    emu_lss_list,
    emu_cmb_list,
    emu_cmb_lensing_list,
    filename_config,
    lims=None,
    gaussian_priors=[],
    gauss_mean=np.zeros(10),
    gauss_sigma=np.ones(10),
    n_walkers=512,
    kw_emcee={"progress": True},
    n_cpus=1,
    x0="likelihood",
    cosmology_params_fixed={},
    bump_vector=None,
    nuisance_params=False,
    indices_list=None,
    compression_vectors_list=None,
    compression_vectors_list_lss_full=None,
    compression_vectors_list_lensing=None,
    lognorm_list=None,
    inv_C_cmb=None,
    dv_cmb=None,
    bl_min_cmb=None,
    bl_max_cmb=None,
    binw_cmb=None,
    lss_cosmopower=False,
    inv_C_gauss=None,
    compression_vectors_list_cmb = None,
    kappa_indices_list=[],
    indices_list_cmb = None,
    lens_data = None,
    cmb_experiment='planck',
    m_bias_indices_list = None,
    act_data = None,
):
    n_samples = int(n_samples)

    if isinstance(filename_config, str):
        conf = load_config(filename_config)
    else:
        conf = filename_config

    if nuisance_params:
        params_start = get_params_grid(
            conf=conf,
            n_samples=n_walkers,
            n_gridpoints=n_walkers,
            first=0,
            sequence="sobol",
            nuisance_params=True,
        )
    else:
        params_start = get_params_grid(
            conf=conf,
            n_samples=n_walkers,
            n_gridpoints=n_walkers,
            first=0,
            sequence="sobol",
        )

    mcmc_start_x0 = get_mcmc_start(params_start, cosmology_params_fixed)
    LOGGER.info("MCMC starting point defined")

    if lims is None:
        lims = get_param_lims(
            conf["variable_cosmology_parameters"], cosmology_params_fixed
        )

    dtype = []
    for p in params_start.dtype.descr:
        if p[0] not in cosmology_params_fixed.keys():
            dtype.append(p)

    samples, lnprobs = _run_mcmc(
        n_samples=n_samples,
        emu_lss_list=emu_lss_list,
        emu_cmb_list=emu_cmb_list,
        emu_cmb_lensing_list=emu_cmb_lensing_list,
        y_obs=cl_obs,
        C=C,
        x0=mcmc_start_x0,
        lims=lims,
        gaussian_priors=gaussian_priors,
        gauss_mean=gauss_mean,
        gauss_sigma=gauss_sigma,
        kw_emcee=kw_emcee,
        indices_list=indices_list,
        compression_vectors_list=compression_vectors_list,
        compression_vectors_list_lss_full=compression_vectors_list_lss_full,
        compression_vectors_list_lensing=compression_vectors_list_lensing,
        bump_vector=bump_vector,
        lognorm_list=lognorm_list,
        inv_C_cmb=inv_C_cmb,
        dv_cmb=dv_cmb,
        bl_min_cmb=bl_min_cmb,
        bl_max_cmb=bl_max_cmb,
        binw_cmb=binw_cmb,
        dtype=dtype,
        cosmology_params_fixed=cosmology_params_fixed,
        nuisance_params=nuisance_params,
        lss_cosmopower=lss_cosmopower,
        inv_C_gauss=inv_C_gauss,
        compression_vectors_list_cmb = compression_vectors_list_cmb,
        kappa_indices_list=kappa_indices_list,
        indices_list_cmb = indices_list_cmb,
        lens_data = lens_data,
        cmb_experiment= cmb_experiment,
        m_bias_indices_list = m_bias_indices_list,
        act_data = act_data,
    )

    params_chain = np.empty(len(samples), dtype=dtype)
    for i, c in enumerate(params_chain.dtype.names):
        params_chain[c] = samples[:, i]

    return params_chain, lnprobs


def get_mcmc_start(params_start, cosmology_params_fixed, nuisance_params=None):
    par = np.array(params_start.dtype.names)
    fix = list(cosmology_params_fixed.keys())
    var = ~np.isin(par, fix)

    print('COSMO PARAMS FIXED:', fix)
    print('VAR:', par[var])

    mcmc_start = np.array([params_start[c] for c in par[var]]).T
    return mcmc_start


def _run_mcmc(
    n_samples,
    emu_lss_list,
    emu_cmb_list,
    emu_cmb_lensing_list,
    y_obs,
    C,
    x0,
    lims,
    gaussian_priors=[],
    gauss_mean=np.zeros(10),
    gauss_sigma=np.ones(10),
    ells=None,
    kw_emcee={},
    indices_list=None,
    compression_vectors_list=None,
    compression_vectors_list_lss_full=None,
    compression_vectors_list_lensing=None,
    bump_vector=None,
    lognorm_list=None,
    inv_C_cmb=None,
    dv_cmb=None,
    bl_min_cmb=None,
    bl_max_cmb=None,
    binw_cmb=None,
    dtype=None,
    cosmology_params_fixed={},
    nuisance_params=False,
    inv_C_gauss=None,
    lss_cosmopower=False,
    compression_vectors_list_cmb = None,
    kappa_indices_list=[],
    indices_list_cmb = None,
    lens_data = None,
    cmb_experiment='planck',
    m_bias_indices_list = None,
    act_data = None,
):
    def select_indices(arr, indices_list):
        # Select emulator output based on series of indices
        new_arr = arr[:, indices_list]

        return new_arr

    def convert_om_to_ocdm(emu_dict):
        # Note omega_m is Capital Om!
        # Specific to AR set-up where m_nu_tot = 3*m_ncdm
        omega_m = np.array(emu_dict["omega_m"])
        omega_b = np.array(emu_dict["omega_b"])
        h = np.array(emu_dict["h"])
        m_nu = np.array(emu_dict["m_nu"])

        # Om = Onu + Oc + Ob
        o_nu = m_nu * 3 / 93.14

        omega_cdm = (omega_m*h*h) - o_nu - omega_b
        emu_dict["omega_cdm"] = omega_cdm

        return emu_dict

    def convert_S8_ocdm_to_sigma8_Om(emu_dict):
        omega_b = np.array(emu_dict["omega_b"])
        h = np.array(emu_dict["h"])
        m_nu = np.array(emu_dict["m_nu"])
        o_nu = m_nu * 3 / 93.14

        S8 = np.array(emu_dict["S8"])
        omega_cdm = np.array(emu_dict["omega_cdm"])

        omega_m = (omega_cdm+omega_b+o_nu)/h/h
        sigma8 = S8*np.sqrt(0.3/omega_m)

        emu_dict["omega_m"] = omega_m
        emu_dict["sigma8"] = sigma8

        return emu_dict



    def transform_to_dict(p, dtype, nuisance_params=None):
        # put the p in the emcee sampler into cosmopower dictionary form
        params_dict = {}

        for i, name in enumerate(dtype):
            params_dict[name[0]] = p[:, i].tolist()

        for key in cosmology_params_fixed.keys():
            params_dict[key] = (
                np.ones((p[:, 0].shape[0],)) * cosmology_params_fixed[key]
            )

        return params_dict

    """
    PLANCK LIKELIHOOD FUNCTIONS
    """

    def get_binned_D_from_theory_Cls(ell, Cl, lmin_list, lmax_list):
        # convert from C to D=l(l+1)C_l/2pi, average in bin, convert back to C
        # loops through lmin and lmax
        # ell 2-29

        D_fac = ell * (ell + 1) / (2 * np.pi)
        Dl = D_fac * Cl

        Dl_bin = np.zeros((Cl.shape[0], len(lmin_list)))

        for i, lmin in enumerate(lmin_list):
            lmax = lmax_list[i]
            Dl_bin[:, i] = np.mean(Dl[:, int(lmin) - 2 : int(lmax) - 2 + 1], axis=1)

        return Dl_bin

    def lognormal(x, mu, sig, loc=0):

        LN = (
            1
            / ((x - loc) * sig * np.sqrt(2 * np.pi))
            * np.exp(-((np.log(x - loc) - mu) ** 2) / (2 * sig**2))
        )

        return LN

    def planck_lowE_binned_loglike(
        Cl_theory, mu_LN_EE, sig_LN_EE, loc_LN_EE, lmin_list_EE, lmax_list_EE
    ):
        # Cl_theory is a numpy array of Cl_EE from ell=2-30
        ell = np.arange(2, 30)
        Dl_theory_bin = get_binned_D_from_theory_Cls(
            ell, Cl_theory, lmin_list_EE, lmax_list_EE
        )
        loglike = np.zeros((Cl_theory.shape[0],), dtype=np.float64)

        for i in range(Dl_theory_bin.shape[1]):
            D = Dl_theory_bin[:, i]
            like_real = lognormal(D, mu_LN_EE[i], sig_LN_EE[i], loc_LN_EE[i])
            loglike += np.log(like_real)
        return loglike

    def planck_lowT_binned_loglike(
        Cl_theory, mu_LN_TT, sig_LN_TT, lmin_list_TT, lmax_list_TT
    ):
        # Cl_theory is a numpy array of Cl_TT from ell=2-30
        ell = np.arange(2, 30)
        Dl_theory_bin = get_binned_D_from_theory_Cls(
            ell, Cl_theory, lmin_list_TT, lmax_list_TT
        )
        loglike = np.zeros((Cl_theory.shape[0],), dtype=np.float64)
        for i in range(Dl_theory_bin.shape[1]):
            D = Dl_theory_bin[:, i]
            like_real = lognormal(D, mu_LN_TT[i], sig_LN_TT[i])
            loglike += np.log(like_real)
        return loglike

    def log_like(
        p,
        y_obs,
        inv_C,
        lims,
        gaussian_priors,
        gauss_mean,
        gauss_sigma,
        emu_lss_list=None,
        emu_cmb_list=None,
        emu_cmb_lensing_list=None,
        ells=None,
        indices_list=None,
        compression_vectors_list=None,
        compression_vectors_list_lss_full=None,
        compression_vectors_list_lensing=None,
        bump_vector=None,
        lognorm_list=None,
        inv_C_cmb=None,
        dv_cmb=None,
        bl_min_cmb=None,
        bl_max_cmb=None,
        binw_cmb=None,
        nuisance_params=False,
        inv_C_gauss=None,
        lss_cosmopower=False,
        compression_vectors_list_cmb = None,
        kappa_indices_list=[],
        indices_list_cmb = None,
        lens_data = None,
        cmb_experiment='planck',
        m_bias_indices_list = None,
        act_data = None, 
    ):
        def log_gauss(x, mu, sigma, inv_C_gauss=None):
            if (inv_C_gauss is not None) and (x.shape[1] > 1):
                diff_gauss = x - np.array(mu).T

                return -0.5 * (diff_gauss.dot(inv_C_gauss) * diff_gauss).sum(axis=1)

            else:
                return (
                    -0.5 * (x - mu) ** 2 / sigma**2
                )  # + np.log(1.0/(np.sqrt(2*np.pi)*sigma))

        like = np.zeros(p.shape[0])

        within_bounds = np.logical_and(p > lims[:, 0], p < lims[:, 1])
        select = np.all(within_bounds, axis=1)

        if (emu_lss_list is None) and (emu_cmb_list is None) and (emu_cmb_lensing_list is None):
            raise ("you must provide at least one emulator")

        # Normal comsopower syntax
        if emu_cmb_lensing_list is not None:
        #For now this is a harcoded Planck CMB lesning likelihood in python. can be updated to include ACT as well
            assert lens_data is not None

            params_dict = transform_to_dict(p[select], dtype, nuisance_params)
            names = [dtype[i][0] for i in range(len(dtype))]
            if "omega_cdm" not in names:
                params_dict = convert_om_to_ocdm(params_dict)

            # Some assertions about the length of this list etc. are missing

            kk_emu_model = emu_cmb_lensing_list[0]
            # sourcing C_ells from CosmoPower

            Cl_kk = np.exp(kk_emu_model.predictions_np(params_dict))*2/np.pi #conversion factor to match up to their units

            #load in all of the data required for the cmb lensing likelihood
            inv_c_lens, cl_hat_lens, bin_mat_lens, corr_lens = lens_data

            #Now apply the binning matrix and get the lkl make sure to match the factors of 2pi with
            #https://arxiv.org/abs/1807.06210
            #corr_lens is the constant term that is the second half of question 29

            # Define the minimum and maximum ell values to be binned

            ell_min = 2
            ell_max = 2500

            cl_binned = Cl_kk[:, ell_min-2:ell_max-2+1] #note that ellmin=2 for this emulator
            cl_binned = np.dot(bin_mat_lens, cl_binned.T).T
            cl_theory_cmbkk = cl_binned - corr_lens #remove the lensing correction following lensing paper- make sure this needs to be removed not added


            #Now compute the LKL

            if emu_lss_list is None:
                if compression_vectors_list_lensing is not None:
                    cl_theory_cmbkk = np.dot(cl_theory_cmbkk, compression_vectors_list_lensing[0])
                #i.e. we want to use kk alone without any xcorrs
                diff = cl_hat_lens - cl_theory_cmbkk

                chi2 = np.matmul(inv_c_lens, np.transpose(diff))
                chi2 = np.matmul(diff, chi2)

                chi2 = np.diag(chi2)

                loglkl = -0.5 * chi2

                like[select] += loglkl


        if emu_lss_list is not None:
            for i, emulator in enumerate(emu_lss_list):
                if i == 0:
                    if lss_cosmopower:
                        params_dict = transform_to_dict(
                            p[select], dtype, nuisance_params
                        )
                        names = [dtype[i][0] for i in range(len(dtype))]
                        if "omega_m" not in names: #Assume WL sampling here!
                            params_dict = convert_S8_ocdm_to_sigma8_Om(params_dict)

                        y_theory = np.exp(emulator.predictions_np(params_dict))

                        if indices_list is not None:
                            cls_emu_out = select_indices(y_theory, indices_list[i])
                        else:
                            cls_emu_out = y_theory

                    else:
                        y_theory = emulator(p[select])
                        if indices_list is not None:
                            cls_emu_out = select_indices(y_theory, indices_list[i])

                        else:
                            cls_emu_out = y_theory

                    if bump_vector is not None:
                        cls_emu_out -= 2 * np.abs(bump_vector[i])

                    if compression_vectors_list is not None:
                        cls_emu_out = np.dot(cls_emu_out, compression_vectors_list[i])

                else:
                    y_theory = np.exp(emulator.predictions_np(params_dict))

                    if indices_list is not None:
                        cls_emu = select_indices(y_theory, indices_list[i])
                    else:
                        cls_emu = y_theory

                    if bump_vector is not None:
                        cls_emu -= 2 * np.abs(bump_vector[i])

                    if compression_vectors_list is not None:
                        cls_emu = np.dot(cls_emu, compression_vectors_list[i])

                    cls_emu_out = np.hstack((cls_emu_out, cls_emu))

            if len(kappa_indices_list) != 0:
                # m_k nuisance parameter for the xcorrs of the cmb lensing
                cls_emu_out[:, kappa_indices_list] / np.array(params_dict["m_k"])[
                    :, None
                ]

            if m_bias_indices_list is not None:
                #loop over the 5 bins of the KiDS data and add an m_bias
                for i in range(5):
                    cls_emu_out[:, m_bias_indices_list[i]] * (1+params_dict[f"m{i}"])
                    #for the autos multiply by the square
                    auto_indices = m_bias_indices_list[i][i*8:(i+1)*8]
                    cls_emu_out[:, auto_indices] * (1+params_dict[f"m{i}"])

            if emu_cmb_lensing_list is not None:
                #Now we want cmb auto with the xcorr so add this to the theorey prediction and y_obs!
                #Make sure when making the covaraince matrix that the lensing is in the lower left corner! 
                cls_emu_out = np.hstack((cls_emu_out, cl_theory_cmbkk))

                y_obs = np.hstack((y_obs, cl_hat_lens))

                #If we want the full LSS compression
                if compression_vectors_list_lss_full is not None:
                    y_obs = y_obs[:-9] #now remove the 9 added kk data points as assume this is already in the MOPED compressed DV

            if compression_vectors_list_lss_full is not None:
                cls_emu_out = np.dot(cls_emu_out, compression_vectors_list_lss_full[0])

            diff = cls_emu_out - y_obs

            # print('chi2', -0.5 * (diff.dot(inv_C) * diff).sum(axis=1))

            like[select] += -0.5 * (diff.dot(inv_C) * diff).sum(axis=1)


        if emu_cmb_list is not None:

            if cmb_experiment == 'planck':
                units_factor = 1e12
                # For now this is hardcoded as a Planck CMB likelihood but later want to make this adaptable for ACT as well


                assert inv_C_cmb is not None
                assert dv_cmb is not None

                # defining binning set-up
                nbintt = 215  # 30-2508   #used when getting covariance matrix
                nbinte = 199  # 30-1996
                nbinee = 199  # 30-1996

                # check these!
                plmin = 30
                plmax = 2508
                ellmin = 2

                # transform the p array into an array that is compatible with the cosmopower emulators
                assert (
                    nuisance_params is not False
                ), "planck likelihood requires A_planck nuisance"

                params_dict = transform_to_dict(p[select], dtype, nuisance_params)
                names = [dtype[i][0] for i in range(len(dtype))]
                if "omega_cdm" not in names:
                    params_dict = convert_om_to_ocdm(params_dict)

                # Some assertions about the length of this list etc. are missing

                tt_emu_model, te_emu_model, ee_emu_model = emu_cmb_list
                # sourcing C_ells from CosmoPower

                Cltt = np.exp(tt_emu_model.predictions_np(params_dict)) * units_factor
                Clte = te_emu_model.predictions_np(params_dict) * units_factor
                Clee = np.exp(ee_emu_model.predictions_np(params_dict)) * units_factor

                spectra_list = ['tt', 'te', 'ee']

                blmin = bl_min_cmb
                blmax = bl_max_cmb
                bin_w = binw_cmb

                begintt = blmin + plmin - ellmin
                endtt = blmax + plmin + 1 - ellmin
                beginte = blmin + plmin - ellmin
                endte = blmax + plmin + 1 - ellmin
                beginee = blmin + plmin - ellmin
                endee = blmax + plmin + 1 - ellmin

                beginwtt = blmin
                endwtt = blmax + 1
                beginwte = blmin
                endwte = blmax + 1
                beginwee = blmin
                endwee = blmax + 1

                indicestt = []
                windowtt = []
                indices_reptt = []
                indiceste = []
                windowte = []
                indices_repte = []
                indicesee = []
                windowee = []
                indices_repee = []

                for i in range(nbintt):
                    idxreptt = np.repeat(i, len(np.arange(begintt[i], endtt[i])))
                    indicestt.append(np.arange(begintt[i], endtt[i]))
                    windowtt.append(bin_w[beginwtt[i] : endwtt[i]])
                    indices_reptt.append(idxreptt)
                flat_list = [item for sublist in indices_reptt for item in sublist]
                indices_reptt = np.array(flat_list)
                flat_list = [item for sublist in indicestt for item in sublist]
                indicestt = np.array(flat_list)
                flat_list = [item for sublist in windowtt for item in sublist]
                windowtt = np.array(flat_list)

                for i in range(nbinte):
                    idxrepte = np.repeat(nbintt + i, len(np.arange(beginte[i], endte[i])))
                    indiceste.append(plmax - 1 + np.arange(beginte[i], endte[i]))
                    windowte.append(bin_w[beginwte[i] : endwte[i]])
                    indices_repte.append(idxrepte)
                flat_list = [item for sublist in indices_repte for item in sublist]
                indices_repte = np.array(flat_list)
                flat_list = [item for sublist in indiceste for item in sublist]
                indiceste = np.array(flat_list)
                flat_list = [item for sublist in windowte for item in sublist]
                windowte = np.array(flat_list)

                for i in range(nbinee):
                    idxrepee = np.repeat(
                        nbintt + nbinte + i, len(np.arange(beginee[i], endee[i]))
                    )
                    indicesee.append(
                        plmax - 1 + plmax - 1 + np.arange(beginee[i], endee[i])
                    )
                    windowee.append(bin_w[beginwee[i] : endwee[i]])
                    indices_repee.append(idxrepee)
                flat_list = [item for sublist in indices_repee for item in sublist]
                indices_repee = np.array(flat_list)
                flat_list = [item for sublist in indicesee for item in sublist]
                indicesee = np.array(flat_list)
                flat_list = [item for sublist in windowee for item in sublist]
                windowee = np.array(flat_list)

                indices_rep = np.concatenate(
                    [indices_reptt, indices_repte, indices_repee], axis=0
                )
                indices = np.concatenate([indicestt, indiceste, indicesee], axis=0)

                window_ttteee = np.hstack((windowtt, windowte, windowee))

                Cl = np.hstack((Cltt, Clte, Clee))

                Cl_bin = np.array(
                    [
                        np.bincount(indices_rep, weights=Cl[index, indices] * window_ttteee)
                        for index in range(Cl.shape[0])
                    ]
                ) / (np.array(params_dict["A_planck"])[:, None] ** 2)


                if indices_list_cmb is not None:
                    Cl_bin = select_indices(Cl_bin, indices_list_cmb[0])

                if compression_vectors_list_cmb is not None:
                    Cl_bin = np.dot(Cl_bin, compression_vectors_list_cmb[0])


                diff = dv_cmb - Cl_bin
                chi2 = np.matmul(inv_C_cmb, np.transpose(diff))
                chi2 = np.matmul(diff, chi2)
                chi2 = np.diag(chi2)

                # print('chi2', chi2)

                loglkl = -0.5 * chi2



            like[select] += loglkl

            test_lognorm = False

            if lognorm_list is not None:

                for i in range(len(lognorm_list)):
                    spec, data_lowl = lognorm_list[i]
                    assert data_lowl is not None
                    (
                        lmin_list_EE,
                        lmax_list_EE,
                        mu_LN_EE,
                        sig_LN_EE,
                        loc_LN_EE,
                        lmin_list_TT,
                        lmax_list_TT,
                        mu_LN_TT,
                        sig_LN_TT,
                    ) = data_lowl

                    if spec == "ee":
                        Cls_ee_2_29 = Clee[:, 0:28]

                        if test_lognorm:
                            Cls_ee_2_29[-1] = cl_ee_test

                        lognorm_ee = planck_lowE_binned_loglike(
                            Cls_ee_2_29,
                            mu_LN_EE,
                            sig_LN_EE,
                            loc_LN_EE,
                            lmin_list_EE,
                            lmax_list_EE,
                        )

                    elif spec == "tt":
                        Cls_tt_2_29 = Cltt[:, 0:28]

                        if test_lognorm:
                            Cls_tt_2_29[-1] = cl_tt_test

                        lognorm_tt = planck_lowT_binned_loglike(
                            Cls_tt_2_29, mu_LN_TT, sig_LN_TT, lmin_list_TT, lmax_list_TT
                        )

                    else:
                        raise (
                            Exception("ONLY PLANCK LOW TT AND PLANCK LOW EE supported")
                        )

        gaussian_add = np.zeros_like(like[select])

        for i, g in enumerate(gaussian_priors):
            if isinstance(g, int):
                # note assume no inv C gauss in this case!
                gaussian_add += log_gauss(
                    p[select, g], gauss_mean[i], gauss_sigma[i], inv_C_gauss=None
                )

            else:
                # note assume list of indices and gaussian covmat in this case
                gaussian_add += log_gauss(
                    p[select, g[0] : g[1]], gauss_mean[i], gauss_sigma[i], inv_C_gauss
                )

        like[select] += gaussian_add

        # Set the values outside if the prior boundaries to zero
        like[~select] = -np.inf

        return like

    n_walkers, n_par = x0.shape

    if C is not None:
        inv_C = np.linalg.inv(C)

    else:
        inv_C = None
    import emcee

    kwargs = {
        "emu_lss_list": emu_lss_list,
        "emu_cmb_list": emu_cmb_list,
        "emu_cmb_lensing_list": emu_cmb_lensing_list,
        "y_obs": y_obs,
        "inv_C": inv_C,
        "lims": lims,
        "gaussian_priors": gaussian_priors,
        "gauss_mean": gauss_mean,
        "gauss_sigma": gauss_sigma,
        "indices_list": indices_list,
        "compression_vectors_list": compression_vectors_list,
        "compression_vectors_list_lss_full":compression_vectors_list_lss_full,
        "bump_vector": bump_vector,
        "lognorm_list": lognorm_list,
        "inv_C_cmb": inv_C_cmb,
        "dv_cmb": dv_cmb,
        "bl_min_cmb": bl_min_cmb,
        "bl_max_cmb": bl_max_cmb,
        "binw_cmb": binw_cmb,
        "nuisance_params": nuisance_params,
        "lss_cosmopower": lss_cosmopower,
        "inv_C_gauss": inv_C_gauss,
        "kappa_indices_list": kappa_indices_list,
        "compression_vectors_list_cmb": compression_vectors_list_cmb,
        "compression_vectors_list_lensing": compression_vectors_list_lensing,
        "indices_list_cmb": indices_list_cmb,
        "lens_data": lens_data,
        "cmb_experiment": cmb_experiment,
        "m_bias_indices_list": m_bias_indices_list,
        "act_data": act_data
    }

    sampler = emcee.EnsembleSampler(
        nwalkers=n_walkers,
        ndim=n_par,
        log_prob_fn=log_like,
        vectorize=True,
        kwargs=kwargs,
    )
    state = sampler.run_mcmc(initial_state=x0, nsteps=10, **kw_emcee)
    sampler.reset()
    try:
        sampler.run_mcmc(state, nsteps=int(np.ceil(n_samples / n_walkers)), **kw_emcee)
    except Exception as err:
        LOGGER.error("Emcee error err={}".format(err))
        LOGGER.error("The chain is probably fine anyway.")

    samples = sampler.get_chain(flat=True)
    lnprobs = sampler.get_log_prob(flat=True)
    samples = samples[:n_samples, :]
    lnprobs = lnprobs[:n_samples]

    return samples, lnprobs

