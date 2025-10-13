<h1>
<p align="center">
  <br>3D Tissue Map Tools
</h1>
<p align="center">
    <span>a python toolkit to create 3D Tissue Maps</span>
</p>

🚧The library is still under development, breaking changes may occur! 🚧

`tissue-map-tools` is a python toolkit to create scalable 3D Tissue Maps from 3D OME-NGFF volumes and segmentation masks, and 3D SpatialData objects.

- **Fast** ⚡, building upon scalable and adaptive data formats like OME-NGFF and the Neuroglancer Precomputed Format, the mesh representation is fast and reliable.
- **Interactive**, users can view results of the toolkit from the browser using [Vitessce](https://vitessce.io).
- **Adaptive**, users can adapt parameters like the sharding, chunking and spatial indexing factors to their needs.
- **Reproducible**, our pipeline is fully deterministic delivering the same results for every run.

## Contact us

Feedback or collaborations are very welcome! Please open a GitHub issue or contact us via direct message in the [scverse Zulip](https://scverse.zulipchat.com/).

## Install

```sh
pip install tissue-map-tools
```

## Usage

Please see the `examples` folder.

## License

The project is licensed under BSD 3-Clause License. Please note that one of the optional dependencies, `igneous-pipeline`, is licensed under GPL-3.0. This implies that if you use `igneous-pipeline` as part of your workflow, you are required to comply with both the terms of the BSD 3-Clause License and the GPL-3.0 license.

The code from `igneous-pipeline` is used for certain meshing operations and exposed in a detached module `igneous_converter`. If GPL-3.0 is not compatible with your use case, you can still use the rest of the `tissue-map-tools` package without any restrictions, and replace the meshing functionality with your own implementation.

## Development

Editable install

```sh
uv venv
source .venv/bin/activate
# install with specific dependency groups
uv sync --group examples --group dev --group test
# note: calling `uv sync` is equivalent to the above since
# `tool.uv.default-groups = ["examples", "dev", "test"]` in `pyproject.toml`
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
