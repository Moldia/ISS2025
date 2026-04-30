The code in this repository is a major update of the original pipeline described in Lee et al., 2024.

The original implementation can be found at https://github.com/Moldia/Lee_2023. However, a substantial update was required to move away from outdated and unmaintained dependencies, improve robustness, and incorporate new features.

This repository is therefore a heavily modified and modernized fork of the original code. The Lee_2023 repository is maintained in a frozen state for reproducibility purposes, but users are strongly encouraged to use this updated version.

---

## A quick glance at the updates

### General

The entire codebase has been updated to run without relying on the default Anaconda channels. This change was motivated by licensing restrictions in some institutions that limit the use of Anaconda.

The pipeline now runs primarily on packages available via **conda-forge** (with the exception of the deconvolution module, which requires NVIDIA packages — NVIDIA channels are supported).

Installation `.yml` files have been simplified by reducing strict version pinning wherever possible, allowing the dependency resolver to handle compatible versions. Only packages with strict requirements remain pinned.

All modules have been updated to be compatible with recent package versions. The codebase has been refactored to handle deprecated functionality (for example, replacing `pandas.append` and similar outdated patterns).

The example notebooks have also been reviewed and corrected. Previously, small issues (such as typos, incorrect imports, or inconsistent assumptions) could lead to subtle analysis errors. These have been fixed to improve clarity and reliability.

---

The sections below provide more detailed information about the individual modules.

# preprocessing module:

The preprocessing module is now able to read imaging data from multiple microscope formats, including:

- Leica **autosaved TIFF files**
- Leica **exported TIFF files**
- Leica **`.lif` files**
- Zeiss **`.czi` files**
- Nikon **`.nd2` files**

Refer to the preprocessing notebook for detailed instructions on how to run the preprocessing functions for each format.

The pipeline automatically parses metadata and organizes data into regions, cycles, tiles, and channels, ensuring consistent downstream processing regardless of input format.

In addition, the preprocessing pipeline has been improved in terms of robustness and transparency:

- Intermediate steps (deconvolution, stitching, retiling) are more strictly validated to avoid silent errors.
- Clear logging is provided throughout the pipeline to track processing steps.
- Retiled tile coordinate CSVs are generated consistently and represent **pixel-based image coordinates (not microns)**.

This design ensures that:

- image geometry remains internally consistent during preprocessing,
- stitching and tiling are not affected by scaling errors,
- physical units (e.g. microns) are handled explicitly at later stages of the ISS pipeline.

# deconvolution module:

The previous deconvolution module based on **Flowdec** has been deprecated and replaced with a new implementation based on **RedLionFish** (https://github.com/rosalindfranklininstitute/RedLionfish).

This change was necessary because Flowdec is no longer actively maintained and depends on outdated versions of TensorFlow, leading to significant compatibility issues with modern environments.

Deconvolution is now **integrated directly into the preprocessing pipeline**, allowing it to be applied seamlessly as part of the standard workflow.

Please refer to the preprocessing notebook for usage details.

# decoding module:

The major update in the decoding module is the introduction of the **dense mode**.

In this modality, spot detection is performed independently for each channel, rather than using a pseudoanchor image (maximum projection across channels) as in the standard mode. While this significantly increases computation time, it improves detection performance in **crowded or high-density samples**, where pseudoanchor-based detection can underestimate spot counts.

Please refer to the example notebooks for details and recommended usage.

A secondary quality metric has also been introduced. This metric reports the ratio between the intensity of the primary ("true") channel and the second strongest channel for each spot.  

This provides additional information beyond standard QC scores, helping to distinguish between:
- high background signal across multiple channels, and  
- true signal overlap (e.g. co-localization or optical bleed-through).

Together, these updates improve both the **sensitivity** of spot detection and the **interpretability** of decoding quality.

# postprocessing module:

The postprocessing module has been updated to better leverage modern hardware and improve usability.

Cellpose now defaults to **GPU execution whenever available**, significantly improving performance compared to the previous CPU-only configuration.

The notebooks have been reorganized into a more modular and user-friendly structure:

- separate notebooks are provided for each segmentation method (**Cellpose** and **StarDist**),
- a dedicated notebook allows visualization of segmentation masks over the DAPI channel,
- support for downstream analysis with **pciSeq** has been added

This restructuring makes the workflow clearer, more flexible, and easier to adapt to different analysis needs.

# probe design module:

This module has been removed from the current repository as it was difficult to maintain in its previous form and required a substantial redesign.

It is currently undergoing a complete rewrite. The new version is planned to be available either as a **web-based tool** or as a **locally deployable Docker container**.

In the meantime, users can refer to the original implementation available in the Lee_2023 repository.

# CARE module

The CARE module has been updated and is now supported through a dedicated workflow.

Three notebooks are provided to guide the full process:

- data generation  
- training  
- prediction  

A pre-trained CARE model is provided. While this model performs well and offers a good speed/quality trade-off compared to classical deconvolution, users are **strongly encouraged to train their own model**, as performance depends on microscope setup and sample characteristics.

In the current workflow, CARE-based denoising is applied **after preprocessing (without deconvolution)** and operates directly on the retiled images.


# What is missing

**THE MANUAL HAS NOT BEEN FULLY UPDATED YET**

The notebooks have been significantly improved and are now clearer and more self-contained. In most cases, users should be able to follow the workflow using the notebooks together with the existing manual.

We are also working on integrating additional deconvolution methods (e.g. Deconwolf), which will be added in future updates.

There are still several useful workflows that are not yet fully integrated into the repository, particularly for downstream analysis and edge-case handling. These include:

- notebooks for generating AnnData objects after Ilastik segmentation  
- notebooks for protein / EdU quantification and thresholding  
- notebooks for segmentation-free approaches  
- notebooks for spatial domain analysis (e.g. Banksy)  

Contributions of such notebooks and workflows are very welcome.

---

Please test the code and report any issues via GitHub.