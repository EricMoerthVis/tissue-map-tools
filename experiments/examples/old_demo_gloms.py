# a newer version of this example, using the new code, is available in examples/gloms/gloms.py

from experiments.mesh_operations import get_meshes_ng
from pathlib import Path
import subprocess
from dirhash import dirhash
import hashlib
import json
import dask.array
import numpy as np
from cloudvolume.dask import to_cloudvolume
from tissue_map_tools.view import view_precomputed_in_neuroglancer

current_path = Path(__file__)
assert str(current_path).endswith("examples/old_demo_gloms.py")
out_path = current_path.parent.parent.parent / "out"
out_path.mkdir(parents=True, exist_ok=True)
multires_path = out_path / "gloms_multires"

##
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

# cleanup previous runs
if multires_path.exists():
    print('please run: "rm -rf out/multires" to remove the files from the previous run')
    # os.system(f"rm -rf {multires_path}")
get_meshes_ng(
    mask_path=str(unzipped_path / "0"),
    out_path=str(multires_path),
    csv_out=str(multires_path / "stats_entity_name.csv"),
    entity_name="entity_name",
    smoothing=5,
    test=True,
)
##
to_cloudvolume(
    cloudpath=str(multires_path), arr=dask.array.from_array(np.array([[[]]]))
)
with open(str(multires_path / "info"), "r") as f:
    d = json.load(f)
    d["mesh"] = "multires"
with open(str(multires_path / "info"), "w") as f:
    json.dump(d, f, indent=4)


view_precomputed_in_neuroglancer(str(multires_path))

# tmt.sub_volume_analysis(
#     mask_path=str(unzipped_path / "0"),
#     raw_path=str(
#         unzipped_path / "0"
#     ),  # in this example we use the same path for raw and mask
#     ome_xml_path=str(unzipped_path / "OME" / "METADATA.ome.xml"),
#     csv_out=str(out_path / "meshes/stats_entity_name.csv"),
#     mask_generation_res="0",
# )
