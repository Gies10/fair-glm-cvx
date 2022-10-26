
import numpy as np
from models.agarwal import ReductionsApproach
from models.fair_glm_cg import FairGeneralizedLinearModel
from models.donini import LinearFERM
from models.bechavod import SquaredDifferenceFairLogistic
from models.zafar import FairnessConstraintModel
from models.zafar import DisparateMistreatmentModel
from models.berk import ConvexFrameworkModel
from models.perez import HSICLinearRegression
from models.oneto import GeneralFairERM
from models.svm import LinearSVM
from sklearn.model_selection import ParameterGrid


def get_model_instance(model):
    if model == 'FGLM' or model == 'GLM':
        return FairGeneralizedLinearModel
    if model == 'SD':
        return SquaredDifferenceFairLogistic
    if model == 'FC':
        return FairnessConstraintModel
    if model == 'DM':
        return DisparateMistreatmentModel
    if model == 'IF' or model == 'GF':
        return ConvexFrameworkModel
    if model == 'HSIC':
        return HSICLinearRegression
    if model == 'SVM':
        return LinearSVM
    if model == 'FERM':
        return LinearFERM
    if model == 'GFERM':
        return GeneralFairERM
    if model == 'BGL' or model == 'SP':
        return ReductionsApproach


def get_parameter_grid(configs):
    args = dict()
    config = configs['param']
    for param in config:
        if int(config[param][2]) < 1:
            args[param] = [0]
        else:
            if configs['log_exp_grid']:
                args[param] = np.round(np.exp(np.linspace(
                    np.log(float(config[param][0])),
                    np.log(float(config[param][1])),
                    int(config[param][2])
                )), 5)
            else:
                args[param] = np.round(np.linspace(
                    float(config[param][0]),
                    float(config[param][1]),
                    int(config[param][2])
                ), 5)
    return ParameterGrid(args)


def define_models(configs):
    models = dict()
    for config in configs:
        models[config] = (
            get_model_instance(config),
            get_parameter_grid(configs[config])
        )

    return models
