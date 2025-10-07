from omegaconf import OmegaConf, DictConfig
from typing import List, Tuple



def parse_configs(config_path: str) -> Tuple[DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig]:
    """Load a config file and return component sections as DictConfigs."""
    config = OmegaConf.load(config_path)
    rae_config = config.get("stage_1", None)
    stage2_config = config.get("stage_2", None)
    transport_config = config.get("transport", None)
    sampler_config = config.get("sampler", None)
    guidance_config = config.get("guidance", None)
    misc = config.get("misc", None)
    training_config = config.get("training", None)
    return rae_config, stage2_config, transport_config, sampler_config, guidance_config, misc, training_config

def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--time-dist-type", type=str, default="logit-normal_0_1")
    #group.add_argument("--time-dist-shift", type=float, default=1/ 0.14433756729740643)
    group.add_argument("--time-dist-shift-base", type=float, default=4096)
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)

def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument("--sampling-method", type=str, default="dopri5", help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq")
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")

def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sampling-method", type=str, default="euler", choices=["euler", "heun"])
    group.add_argument("--diffusion-form", type=str, default="sigma", \
                        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],\
                        help="form of diffusion coefficient in the SDE")
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument("--last-step", type=none_or_str, default="Mean", choices=[None, "Mean", "Tweedie", "Euler"],\
                        help="form of last step taken in the SDE")
    group.add_argument("--last-step-size", type=float, default=0.04, \
                        help="size of the last step taken")
