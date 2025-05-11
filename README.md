<h1>
<p align="center">
  <br>3D Tissue Map Tools
</h1>
<p align="center">
    <span>a python toolkit to create 3D Tissue Maps</span>
</p>

**tissue-map-tools** is a python toolkit to create scalable 3D Tissue Maps from 3D OME-NGFF Segmentation Masks

- **interactive** users can view results of the toolkit using [Vitessce](https://vitessce.io)
- **fast** ⚡ building upon scalable and adaptive data formats like OME-NGFF the mesh creation is fast and reliable
- **adaptive** users can adapt parameters like the smoothing factor to delover visual pleasing and biological meaningful results
- **reproducible** Our pipeline is fully deterministic delivering the same results for every run

## install

```sh
pip install tissue-map-tools
```

## usage

```python
import tissue_map_tools as tmt

tmt.get_meshes("http://address_of_hosted_zarr_segmentation_mask", "/path_to_output/meshes",
           "/path_to_output/meshes/stats_entity_name.csv", "entity_name", smoothing=5, test=False)

tmt.sub_volume_analysis("http://address_of_hosted_zarr_segmentation_mask",
                    "http://address_of_hosted_raw_volume",
                    ome_xml_path="http://address_of_hosted_ome_xml_file",
                    csv_out="/path_to_output/meshes/stats_entity_name.csv", mask_generation_res='0')

```

## development

Editable install

```sh
uv venv
source .venv/bin/activate
# install with specific dependency groups
uv sync --group examples --group dev
# note: calling `uv sync` is equivalent to the above since
# `tool.uv.default-groups = ["examples", "dev"]` in `pyproject.toml`
```

Adding/removing packages

```sh
uv add <package_name>
uv remove <package_name>
# to add/remove to a group
uv add --group <group_name> <package_name>
# shortcut to add/remove to the dev group
uv add --dev <package_name>
```

Building the package for distribution

```sh
uv build
```

Using `pre-commit`.

```sh
# install
pre-commit install

# pre-commit are run automatically on commit; you can run on request with
pre-commit run --all-files

# to commit without running pre-commit hooks
git commit --no-verify
```
