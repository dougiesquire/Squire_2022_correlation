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
        
        # logger.info("Preparing HadGEM3-GC31-MM data")
        # _ = data.prepare_HadGEM3_GC31_MM(
        #     ["dcppA-hindcast", "dcppB-forecast"],
        #     "Amon",
        #     ["psl"],
        #     "psl_Amon_HadGEM3-GC31-MM_dcpp"
        # )
        # _ = data.prepare_HadGEM3_GC31_MM(
        #     ["historical"],
        #     "Amon",
        #     ["psl"],
        #     "psl_Amon_HadGEM3-GC31-MM_historical"
        # )
        # _ = data.prepare_HadGEM3_GC31_MM(
        #     ["dcppA-hindcast", "dcppB-forecast"],
        #     "Omon",
        #     ["tos"],
        #     "tos_Omon_HadGEM3-GC31-MM_dcpp"
        # )
        # _ = data.prepare_HadGEM3_GC31_MM(
        #     ["historical"],
        #     "Omon",
        #     ["tos"],
        #     "tos_Omon_HadGEM3-GC31-MM_historical"
        # )

        # logger.info("Preparing EC-Earth3 data")
        # _ = data.prepare_EC_Earth3(
        #     ["dcppA-hindcast"],
        #     "Amon",
        #     ["psl"],
        #     "psl_Amon_EC-Earth3_dcpp"
        # )
        # _ = data.prepare_EC_Earth3(
        #     ["historical"],
        #     "Amon",
        #     ["psl"],
        #     "psl_Amon_EC-Earth3_historical"
        # )
        # _ = data.prepare_EC_Earth3(
        #     ["dcppA-hindcast"],
        #     "Omon",
        #     ["tos"],
        #     "tos_Omon_EC-Earth3_dcpp"
        # )
        # _ = data.prepare_EC_Earth3(
        #     ["historical"],
        #     "Omon",
        #     ["tos"],
        #     "tos_Omon_EC-Earth3_historical"
        # )

        # logger.info("Preparing CanESM5 data")
        # _ = data.prepare_CanESM5(
        #     ["dcppA-hindcast", "dcppB-forecast"],
        #     "Amon",
        #     ["psl"],
        #     "psl_Amon_CanESM5_dcpp"
        # )
        # _ = data.prepare_CanESM5(
        #     ["historical"],
        #     "Amon",
        #     ["psl"],
        #     "psl_Amon_CanESM5_historical"
        # )
        # # REQUIRES hugemem NODE
        # _ = data.prepare_CanESM5(
        #     ["dcppA-hindcast", "dcppB-forecast"],
        #     "Omon",
        #     ["tos"],
        #     "tos_Omon_CanESM5_dcpp"
        # )
        # _ = data.prepare_CanESM5(
        #     ["historical"],
        #     "Omon",
        #     ["tos"],
        #     "tos_Omon_CanESM5_historical"
        # )

        # logger.info("Preparing CESM1.1 data")
        # _ = data.prepare_CESM1_1_CAM5_CMIP5(
        #     ["dcppA-hindcast"],
        #     "Amon",
        #     ["psl"],
        #     "psl_Amon_CESM1-1-CAM5-CMIP5_dcpp"
        # )
        # # REQUIRES hugemem NODE
        # _ = data.prepare_CESM1_1_CAM5_CMIP5(
        #     ["dcppA-hindcast"],
        #     "Omon",
        #     ["tos"],
        #     "tos_Omon_CESM1-1-CAM5-CMIP5_dcpp"
        # )
        # # NO HISTORICAL RUN

        # logger.info("Preparing MIROC6 data")
        # _ = data.prepare_MIROC6(
        #     ["dcppA-hindcast"],
        #     "Amon",
        #     ["psl"],
        #     "psl_Amon_MIROC6_dcpp"
        # )
        # _ = data.prepare_MIROC6(
        #     ["historical"],
        #     "Amon",
        #     ["psl"],
        #     "psl_Amon_MIROC6_historical"
        # )
        # _ = data.prepare_MIROC6(
        #     ["dcppA-hindcast"],
        #     "Omon",
        #     ["tos"],
        #     "tos_Omon_MIROC6_dcpp"
        # )
        # _ = data.prepare_MIROC6(
        #     ["historical"],
        #     "Omon",
        #     ["tos"],
        #     "tos_Omon_MIROC6_historical"
        # )

        # logger.info("Preparing MPI-ESM1.2-HR data")
        # _ = data.prepare_MPI_ESM1_2_HR(
        #     ["dcppA-hindcast"],
        #     "Amon",
        #     ["psl"],
        #     "psl_Amon_MPI-ESM1-2-HR_dcpp"
        # )
        # _ = data.prepare_MPI_ESM1_2_HR(
        #     ["historical"],
        #     "Amon",
        #     ["psl"],
        #     "psl_Amon_MPI-ESM1-2-HR_historical"
        # )
        # # REQUIRES hugemem NODE
        # _ = data.prepare_MPI_ESM1_2_HR(
        #     ["dcppA-hindcast"],
        #     "Omon",
        #     ["tos"],
        #     "tos_Omon_MPI-ESM1-2-HR_dcpp"
        # )
        # _ = data.prepare_MPI_ESM1_2_HR(
        #     ["historical"],
        #     "Omon",
        #     ["tos"],
        #     "tos_Omon_MPI-ESM1-2-HR_historical"
        # )

        # logger.info("Preparing IPSL-CM6A-LR data")
        # _ = data.prepare_IPSL_CM6A_LR(
        #     ["dcppA-hindcast"], "Amon", ["psl"], "psl_Amon_IPSL-CM6A-LR_dcpp"
        # )
        # _ = data.prepare_IPSL_CM6A_LR(
        #     ["historical"],
        #     "Amon",
        #     ["psl"],
        #     "psl_Amon_IPSL-CM6A-LR_historical"
        # )
        # REQUIRES hugemem NODE
        # _ = data.prepare_IPSL_CM6A_LR(
        #     ["dcppA-hindcast"], "Omon", ["tos"], "tos_Omon_IPSL-CM6A-LR_dcpp"
        # )
        # _ = data.prepare_IPSL_CM6A_LR(
        #     ["historical"],
        #     "Omon",
        #     ["tos"],
        #     "tos_Omon_IPSL-CM6A-LR_historical"
        # )

        # logger.info("Preparing NorCPM1")
        # _ = data.prepare_NorCPM1(
        #     ["dcppA-hindcast"],
        #     "Amon",
        #     ["psl"],
        #     "psl_Amon_NorCPM1_dcpp"
        # )
        # _ = data.prepare_NorCPM1(
        #     ["historical"],
        #     "Amon",
        #     ["psl"],
        #     "psl_Amon_NorCPM1_historical"
        # )
        # _ = data.prepare_NorCPM1(
        #     ["dcppA-hindcast"],
        #     "Omon",
        #     ["tos"],
        #     "tos_Omon_NorCPM1_dcpp"
        # )
        # _ = data.prepare_NorCPM1(
        #     ["historical"],
        #     "Omon",
        #     ["tos"],
        #     "tos_Omon_NorCPM1_historical"
        # )
        
        # logger.info("Preparing CMCC-CM2-SR5")
        # _ = data.prepare_CMCC_CM2_SR5(
        #     ["dcppA-hindcast", "dcppB-forecast"],
        #     "Amon",
        #     ["psl"],
        #     "psl_Amon_CMCC-CM2-SR5_dcpp"
        # )
        # _ = data.prepare_CMCC_CM2_SR5(
        #     ["historical"],
        #     "Amon",
        #     ["psl"],
        #     "psl_Amon_CMCC-CM2-SR5_historical"
        # )
        # _ = data.prepare_CMCC_CM2_SR5(
        #     ["dcppA-hindcast", "dcppB-forecast"],
        #     "Omon",
        #     ["tos"],
        #     "tos_Omon_CMCC-CM2-SR5_dcpp"
        # )
        # _ = data.prepare_CMCC_CM2_SR5(
        #     ["historical"],
        #     "Omon",
        #     ["tos"],
        #     "tos_Omon_CMCC-CM2-SR5_historical"
        # )
        
        # logger.info("Preparing MRI-ESM2-0")
        # _ = data.prepare_MRI_ESM2_0(
        #     ["dcppA-hindcast"],
        #     "Amon",
        #     ["psl"],
        #     "psl_Amon_MRI-ESM2-0_dcpp"
        # )
        # DATA REQUEST LODGED
        # _ = data.prepare_MRI_ESM2_0(
        #     ["historical"],
        #     "Amon",
        #     ["psl"],
        #     "psl_Amon_MRI-ESM2-0_historical"
        # )
        # _ = data.prepare_MRI_ESM2_0(
        #     ["dcppA-hindcast"],
        #     "Omon",
        #     ["tos"],
        #     "tos_Omon_MRI-ESM2-0_dcpp"
        # )
        # _ = data.prepare_MRI_ESM2_0(
        #     ["historical"],
        #     "Omon",
        #     ["tos"],
        #     "tos_Omon_MRI-ESM2-0_historical"
        # )

        # logger.info("Preparing HadSLP2r data")
        # _ = data.prepare_HadSLP2r()

        # logger.info("Preparing HadISST data")
        # _ = data.prepare_HadISST()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
