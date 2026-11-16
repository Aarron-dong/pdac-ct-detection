# Attribution and data terms

The anatomical two-stage framework follows Alves N, Schuurmans M, Litjens G, Bosma JS, Hermans J, Huisman H. *Fully Automatic Deep Learning Framework for Pancreatic Ductal Adenocarcinoma Detection on Computed Tomography*. Cancers. 2022;14:376. https://doi.org/10.3390/cancers14020376

Upstream implementation: https://github.com/DIAGNijmegen/CE-CT_PDAC_AutomaticDetection_nnUnet. Its Apache 2.0 license is retained in `licenses/Alves_algorithm_APACHE_2_0.txt`. The inference implementation adapts that framework to the networks trained for the accompanying PDAC study.

Architecture construction, preprocessing, augmentation and sliding-window inference use nnU-Net v2. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*. Nature Methods. 2021;18:203–211. https://doi.org/10.1038/s41592-020-01008-z

Upstream: https://github.com/MIC-DKFZ/nnUNet. Its Apache 2.0 license and the licenses of other reference dependencies are retained in `licenses/`. Dependencies are installed separately.

PANORAMA development-data terms are Creative Commons Attribution-NonCommercial 4.0, retained in `licenses/PANORAMA_data_CC_BY_NC_4_0.txt`. The release and associated batches are identified by https://doi.org/10.5281/zenodo.13715870, https://doi.org/10.5281/zenodo.13742336, https://doi.org/10.5281/zenodo.11034011 and https://doi.org/10.5281/zenodo.10999754. Dataset creators: Natalia Alves, Megan Schuurmans, Derya Yakar, Pierpaolo Vendittelli, Geert Litjens, John Hermans and Henkjan Huisman. The release also requests citation of the PANORAMA Study Protocol, https://doi.org/10.5281/zenodo.10599559.

MSD Task07 supplied source initialization data. Antonelli M et al. *The Medical Segmentation Decathlon*. Nature Communications. 2022;13:4128. https://doi.org/10.1038/s41467-022-30695-9. The supplied source license information is retained in `licenses/MSD_LICENSE_SOURCE.txt`.

This repository distributes inference code and project-trained weights. Project code is Apache 2.0; project-trained weights use the separate noncommercial research license. Source images, annotations and independently licensed materials retain their original terms.
