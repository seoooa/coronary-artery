# HaN-Seg: The head and neck organ-at-risk CT & MR segmentation dataset

## Reference
Please cite the following paper, if you are using the HaN-Seg: The head and neck organ-at-risk CT & MR segmentation dataset:
	
	G. Podobnik, P. Strojan, P. Peterlin, B. Ibragimov, T. Vrtovec, "HaN-Seg: The head and neck organ-at-risk CT & MR segmentation dataset", Medical Physics, 2023. https://doi.org/10.1002/mp.16197

	@ARTICLE{HaNSeg_dataset,
		author = {Ga\v{s}per Podobnik, Primo\v{z} Strojan, Primo\v{z} Peterlin, Bulat Ibragimov, Toma\{z} Vrtovec},
		title = {{HaN-Seg}: {T}he head and neck organ-at-risk {CT} \& {MR} segmentation dataset},
		journal = {Medical Physics},
		year = {2023},
		doi = {https://doi.org/10.1002/mp.16197}
	}


## License
[CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/)


## Dataset Characteristics
**The HaN-Seg: Head and Neck Organ-at-Risk CT & MR Segmentation Dataset** is a publicly available dataset of anonymized head and neck (HaN) images of 42 patients that underwent both CT and T1-weighted MR imaging for the purpose of image-guided radiotherapy planning. In addition, the dataset also contains reference segmentations of 30 organs-at-risk (OARs) for CT images in the form of binary segmentation masks, which were obtained by curating manual pixel-wise expert image annotations. 

**A full description of the HaN-Seg dataset can be found [here](https://doi.org/10.1002/mp.16197).**


## Folder Structure
```
HaN-Seg
├── set_1
│	├── case_01
│	│	├── case_01_IMG_CT.nrrd (CT image file)
│	│	├── case_01_IMG_MR_T1.nrrd (T1-weighted MR image file)
│	│	├── case_01_OAR_A_Carotid_L.seg.nrrd (left carotid artery binary segmentation file)
│	│	├── case_01_OAR_A_Carotid_R.seg.nrrd (right carotid artery binary segmentation file)
│	│	├── ...
│	│	└── case_01_OAR_SpinalCord.seg.nrrd (spinal cord binary segmentation file)
│	├── ...
│	├── case_42
│	│	└── ...
│	├── OAR_data.csv (`.csv` file with segmentation availability information, see Chapter 3 of our paper for datails)
│	└── patient_data.csv (`.csv` file with demographic information for all patients)
├── README.md
└── LICENSE (CC BY-ND 4.0)
```


## Managed By
Laboratory of Imaging Technologies,
Faculty of Electrical Engineering,
University of Ljubljana,
Slovenia
