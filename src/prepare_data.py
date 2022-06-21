"CLI for preparing data"

import logging

import tempfile

from dask.distributed import Client

from src import data


def main():
    """
    Spin up a dask cluster and process and save raw data
    """
    logger = logging.getLogger(__name__)

    logger.info("Spinning up a dask cluster")
    local_directory = tempfile.TemporaryDirectory()
    with Client(processes=False, local_directory=local_directory.name):
        # logger.info("Preparing HadGEM3 data")
        # _ = data.prepare_HadGEM3_dcpp_variable("psl", "Amon")
        # _ = data.prepare_HadGEM3_dcpp_variable("tos", "Omon")

        # logger.info("Preparing EC-Earth3 data")
        # _ = data.prepare_EC_Earth3_dcpp_variable("psl", "Amon")
        # _ = data.prepare_EC_Earth3_dcpp_variable("tos", "Omon")
        
        # logger.info("Preparing CanESM5 data")
        # _ = data.prepare_CanESM5_dcpp_variable("psl", "Amon")
        # _ = data.prepare_CanESM5_dcpp_variable("tos", "Omon")

        logger.info("Preparing CESM1.1 data")
        prepare_CESM1_1_CAM5_CMIP5_dcpp_variable("psl", "Amon")
        prepare_CESM1_1_CAM5_CMIP5_dcpp_variable("tos", "Omon")
    
        # logger.info("Preparing HadSLP2r data")
        # _ = data.prepare_HadSLP2r()

        # logger.info("Preparing HadISST data")
        # _ = data.prepare_HadISST()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
