from datetime import date
from pathlib import Path

import h5py
import pytest

from disp_s1 import product
from disp_s1.pge_runconfig import RunConfig

TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_CSLC_FILE = (
    TEST_DATA_DIR
    / "OPERA_L2_CSLC-S1_T087-185683-IW2_20221228T161651Z_20240504T181714Z_S1A_VV_v1.1.h5"  # noqa: E501
)

TEST_OUTPUT_CCSLC_FILE = TEST_DATA_DIR / "compressed_20221228_20230101_20230113.tif"


# Shapely runtime warning
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_create_output_product(
    tmp_path,
    test_data_dir,
    scratch_dir,
):
    # Get the test run scratch directory
    # $ ls scratch/forward/
    # interferograms  log_sas.log  t042_088905_iw1  t042_088906_iw1  unwrapped
    # $ ls scratch/forward/unwrapped/
    # 20221119_20221213.unw.conncomp  20221119_20221213.unw.conncomp.hdr  ...
    unw_dir = scratch_dir / "forward/unwrapped"
    unw_filename = unw_dir / "20221119_20221213.unw.tif"
    conncomp_filename = unw_dir / "20221119_20221213.unw.conncomp"

    ifg_dir = scratch_dir / "forward/interferograms/stitched"
    tcorr_filename = ifg_dir / "tcorr.tif"
    ps_mask_filename = ifg_dir / "ps_mask_looked.tif"
    ifg_corr_filename = ifg_dir / "20221119_20221213.cor"

    cslc_files = sorted(
        (test_data_dir / "delivery_data_small/input_slcs").glob("t*.h5")
    )
    assert len(cslc_files) > 0

    output_name = tmp_path / "20221119_20221213.nc"

    pge_runconfig = RunConfig.from_yaml(
        test_data_dir / "delivery_data_small/config_files/runconfig_forward.yaml"
    )

    product.create_output_product(
        unw_filename=unw_filename,
        conncomp_filename=conncomp_filename,
        tcorr_filename=tcorr_filename,
        ifg_corr_filename=ifg_corr_filename,
        output_name=output_name,
        corrections={},
        ps_mask_filename=ps_mask_filename,
        pge_runconfig=pge_runconfig,
        cslc_files=cslc_files,
    )


def test_create_compressed_slc(tmp_path):
    # OPERA_L2_CSLC-S1_T087-185683-IW2_20221228T161651Z_20240504T181714Z_S1A_VV_v1.1.h5
    # compressed_20221228_20230101_20230113.tif
    date_str = "20221228_20230101_20230113"
    burst_id = "t087_185683_iw2"
    processed_ccslc_file = TEST_OUTPUT_CCSLC_FILE
    comp_slc_dict = {burst_id: [processed_ccslc_file]}

    product.create_compressed_products(
        comp_slc_dict, cslc_file_list=[TEST_CSLC_FILE], output_dir=tmp_path
    )

    expected_name = tmp_path / product.COMPRESSED_SLC_TEMPLATE.format(
        burst_id=burst_id, date_str=date_str
    )
    assert expected_name.exists()
    # Check product structure
    with h5py.File(expected_name) as hf:
        assert hf["/data/VV"].size > 0
        assert "/metadata/orbit" in hf
        assert "/identification/zero_doppler_start_time" in hf
        assert "/metadata/processing_information/input_burst_metadata/wavelength" in hf


class TestGenerationDatetime:
    """A CSLC filename may or may not carry a generation datetime."""

    def test_opera_name_uses_the_generation_datetime(self):
        """The second of the two datetimes an OPERA name carries."""
        from disp_s1.product import _generation_datetime

        name = (
            "OPERA_L2_CSLC-S1_T087-185683-IW2_20221228T161651Z"
            "_20240504T181714Z_S1A_VV_v1.1.h5"
        )
        assert _generation_datetime(Path(name)).date() == date(2024, 5, 4)

    def test_compass_name_falls_back_to_the_acquisition_date(self):
        """Regression: `get_dates(f)[1]` raised IndexError on these names.

        A COMPASS-convention stack (a stripmap one, say) reached product
        creation intact and then failed on the missing second date -- after
        every product had been written but before any was complete.
        """
        from disp_s1.product import _generation_datetime

        got = _generation_datetime(Path("t095_000004_iw1_20161019.h5"))

        assert got.date() == date(2016, 10, 19)

    def test_compressed_slc_name_uses_its_second_date(self):
        from disp_s1.product import _generation_datetime

        got = _generation_datetime(Path("compressed_20170101_20161019_20170101.h5"))

        assert got.date() == date(2016, 10, 19)
