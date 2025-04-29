import tissue_map_tools as tmt
from pathlib import Path
import subprocess
from dirhash import dirhash
import hashlib

current_path = Path(__file__)
assert str(current_path).endswith("examples/demo.py")
out_path = current_path.parent.parent / "out"
out_path.mkdir(parents=True, exist_ok=True)

# download the example data
URL = "https://s3.embl.de/spatialdata/raw_data/20_1_gloms.zip"
CHECKSUM_DOWNLOAD = "7857a41d9d4d2914353c9ad0f4ea4ede"
CHECKSUM_UNZIPPED = "927146f7a8cbfcbf9de047a6e1e71226"

download_path = out_path / Path(URL).name
if (
    not download_path.exists()
    or CHECKSUM_DOWNLOAD != hashlib.md5(download_path.read_bytes()).hexdigest()
):
    subprocess.run(f'curl -o "{download_path}" "{URL}"', shell=True, check=True)

# unzip the downloaded file
unzipped_path = out_path / Path(URL).stem
if not unzipped_path.exists() or CHECKSUM_UNZIPPED != dirhash(unzipped_path, "md5"):
    subprocess.run(
        f'unzip -o "{download_path}" -d "{out_path}"', shell=True, check=True
    )

tmt.get_meshes(
    mask_path=str(unzipped_path / "0"),
    out_path=str(out_path / "meshes"),
    csv_out=str(out_path / "meshes/stats_entity_name.csv"),
    entity_name="entity_name",
    smoothing=5,
    test=False,
)

tmt.sub_volume_analysis(
    mask_path=str(unzipped_path / "0"),
    raw_path=str(
        unzipped_path / "0"
    ),  # in this example we use the same path for raw and mask
    ome_xml_path=str(unzipped_path / "OME" / "METADATA.ome.xml"),
    csv_out=str(out_path / "meshes/stats_entity_name.csv"),
    mask_generation_res="0",
)
