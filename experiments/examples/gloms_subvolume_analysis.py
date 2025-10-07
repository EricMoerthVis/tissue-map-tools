# run examples/gloms/gloms.py before running this example
from pathlib import Path
from experiments.subvolume_analysis import sub_volume_analysis

out_path = Path(__file__).parent.parent.parent / "out"
precomputed_path = out_path / "gloms_precomputed"

##
# download and unzip the data
URL = "https://s3.embl.de/spatialdata/raw_data/20_1_gloms.zip"
unzipped_path = out_path / Path(URL).stem

##
sub_volume_analysis(
    mask_path=str(unzipped_path / "0"),
    raw_path=str(
        unzipped_path / "0"
    ),  # in this example we use the same path for raw and mask
    ome_xml_path=str(unzipped_path / "OME" / "METADATA.ome.xml"),
    csv_out=str(out_path / "meshes/stats_entity_name.csv"),
    mask_generation_res="0",
)
