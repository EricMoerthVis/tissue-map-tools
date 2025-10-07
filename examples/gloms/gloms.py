from pathlib import Path
import subprocess
from dirhash import dirhash
import hashlib
from tissue_map_tools.view import view_precomputed_in_neuroglancer
from tissue_map_tools.igneous_converters import (
    from_ome_zarr_04_raster_to_sharded_precomputed_raster_and_meshes,
)

out_path = Path(__file__).parent.parent.parent / "out"
precomputed_path = out_path / "gloms_precomputed"

##
# download and unzip the data
URL = "https://s3.embl.de/spatialdata/raw_data/20_1_gloms.zip"
CHECKSUM_DOWNLOAD = "7857a41d9d4d2914353c9ad0f4ea4ede"
CHECKSUM_UNZIPPED = "927146f7a8cbfcbf9de047a6e1e71226"
unzipped_path = out_path / Path(URL).stem

##
download_path = out_path / Path(URL).name
if (
    not download_path.exists()
    or CHECKSUM_DOWNLOAD != hashlib.md5(download_path.read_bytes()).hexdigest()
):
    subprocess.run(f'curl -o "{download_path}" "{URL}"', shell=True, check=True)

# unzip the downloaded file
if not unzipped_path.exists() or CHECKSUM_UNZIPPED != dirhash(unzipped_path, "md5"):
    subprocess.run(
        f'unzip -o "{download_path}" -d "{out_path}"', shell=True, check=True
    )
##
from_ome_zarr_04_raster_to_sharded_precomputed_raster_and_meshes(
    ome_zarr_path=str(unzipped_path / "0"),
    precomputed_path=str(precomputed_path),
)

view_precomputed_in_neuroglancer(str(precomputed_path))
