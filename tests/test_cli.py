import os

import pytest

from foundation_ts import cli


def test_cli_download_real_hf(tmp_path):
    if os.getenv("FOUNDATIONTS_RUN_HF_DOWNLOAD") != "1":
        pytest.skip("Set FOUNDATIONTS_RUN_HF_DOWNLOAD=1 to run the real HF download.")

    partitions_env = os.getenv("FOUNDATIONTS_TEST_PARTITIONS")
    if partitions_env:
        partitions = [p.strip() for p in partitions_env.split(",") if p.strip()]
    else:
        partitions = ["other/m4_daily"]

    argv = [
        "data",
        "download",
        "--partitions",
        ",".join(partitions),
        "--time300b-dir",
        str(tmp_path),
    ]
    cli.main(argv)

    for partition in partitions:
        assert (tmp_path / partition).exists()


def test_cli_train_requires_dataset_path():
    with pytest.raises(ValueError, match="Missing required args"):
        cli.main(["train"])
