from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def print_and_check_modeling_in_config(conf):
    LOGGER.info("Modeling choices:")
    for key, value in dict(conf["analysis"]["modelling"]).items():
        if isinstance(value, dict):
            LOGGER.info(f"{key}:")
            for k, v in value.items():
                LOGGER.info(f"  {k} = {v}")
        else:
            LOGGER.info(f"{key} = {value}")

    if conf["analysis"]["modelling"]["degrade_to_grf"]:
        LOGGER.warning(f"Degrading to Gaussian Random Field")

    # lensing
    conf_lensing = conf["analysis"]["modelling"]["WL"]
    if conf_lensing["extended_nla"]:
        assert conf["analysis"]["params"]["ia"]["tatt"] == ["bta"]

    # clustering
    bg_params = conf["analysis"]["params"]["bg"]["linear"]
    conf_clustering = conf["analysis"]["modelling"]["GC"]

    assert not (
        conf_clustering["power_law_biasing"] and conf_clustering["per_bin_biasing"]
    ), "Cannot have both power law and per bin biasing"

    assert conf_clustering["power_law_biasing"] == {"bg", "n_bg"}.issubset(
        bg_params
    ), "Power law biasing is not consistent with bg parameters"

    assert conf_clustering["per_bin_biasing"] == {
        f"bg{i+1}" for i, _ in enumerate(conf["survey"]["GC"]["z_bins"])
    }.issubset(bg_params), "Per bin biasing is not consistent with bg parameters"

    if conf_clustering["stochasticity"]:
        assert conf["analysis"]["params"]["bg"]["stochasticity"] == ["rg"]


def get_indices(tasks):
    """
    Parses the jobids from the tasks string.

    :param tasks: The task string, which will get parsed into the job indices
    :return: A list of the jobids that should be executed
    """
    # parsing a list of indices from the tasks argument

    if '>' in tasks:
        tasks = tasks.split('>')
        start = tasks[0].replace(' ', '')
        stop = tasks[1].replace(' ', '')
        indices = list(range(int(start), int(stop)))
    elif ',' in tasks:
        indices = tasks.split(',')
        indices = list(map(int, indices))
    else:
        try:
            indices = [int(tasks)]
        except ValueError:
            raise ValueError("Tasks argument is not in the correct format!")

    return indices

