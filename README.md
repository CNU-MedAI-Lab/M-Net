# M-Net: MRI Brain Tumor Sequential Segmentation Network via Mesh-Cast

Considered for ICCV 2025. Paper link: 

https://openaccess.thecvf.com/content/ICCV2025/html/Lu_M-Net_MRI_Brain_Tumor_Sequential_Segmentation_Network_via_Mesh-Cast_ICCV_2025_paper.html

This repository contains the official PyTorch implementation of M-Net, a mesh-cast based sequential segmentation framework for medical image segmentation, proposed in our ICCV 2025 submission.

M-Net models inter-slice dependencies in 2D medical image segmentation by introducing a sequence-aware propagation mechanism, enabling effective contextual aggregation without relying on full 3D convolutions. The framework is particularly designed for volumetric medical images such as brain MRI.

## 🔧 Environment
	•	Python ≥ 3.8
	just like VMamba, if you have already solved that, you can also run us.

## Create environment

	conda create -n mnet python=3.8
	conda activate mnet
	pip install -r requirements.txt

## 📁 Repository Structure

	M-Net_code/
	├── data_processing/
	│   ├── cut_MICCAI_for_ordered.py    # Sequence cuting from 3D nii files
	│   └──  cut_MICCAI_for_shuffled.py   # 2D slices 
	├── run/
	│   ├── __init__.py
	│   ├── train_firstTPS_shuffled.py   # TPS first(shuffled)
	│   ├── train_secondTPS_ordered.py   # TPS second(ordered)
	│   ├── test_on_sequence.py          # testing / evaluation script
	│   └── dataset.py                   # Dataset definition
	├── model/
	│   ├── __init__.py
	│   ├── M_Net_Mamba.py               # M-Net with different sequence model
	│   ├── M_Net_xxx.py    
	│   └── ...
	├── utils/
	│   ├── __init__.py
	│   ├── losses.py
	│   ├── metrics.py
	│   └── misc.py
	├── checkpoint/
	├── requirements.txt
	└── README.md

## 📊 Dataset Preparation
Using data_processing to prepare the 2D image (for shuffled), and sequence image (for ordered).
If you don't want to use TPS strategy, you can also use 2D/sequence dataset only. M-Net will treat them to sequence input.


## 🚀 Training

Training is performed using module-based execution, which is required for relative imports.

	cd M-Net_code
	
	TPS first
	python -m run.train_firstTPS_shuffled \
	  --name M_Net_Mamba \
	  --dataset_path /path/to/DatasetRoot \
	  --batch-size 15 \
	  --gpu_device 0
	
	TPS second (will load the checkpoint after TPS first)
	python -m run.train_secondTPS_ordered \
	  --name M_Net_Mamba \
	  --dataset_path /path/to/DatasetRoot \
	  --batch-size 15 \
	  --gpu_device 0

All experiment configurations are automatically saved to:
	
	checkpoint/{name}/args.pkl
	checkpoint/{name}/args.txt



## 🧪 Testing / Evaluation

To evaluate a trained model:
	
	python -m run.test2 \
	  --name M_Net_Mamba \
	  --gpu_device 0

Outputs are saved to:

	checkpoint/{name}/
	├── image/
	├── mask/
	├── result/
	└── testlogall_new.csv


## 📈 Metrics

The following metrics are reported:
	•	Dice Score (WT / TC / ET)
	•	Intersection over Union (IoU)
	•	Sensitivity
	•	Positive Predictive Value (PPV)
	•	Hausdorff Distance (HD)

All metrics are computed slice-wise and aggregated sequence-wise, following the evaluation protocol described in the paper.

## 🧠 Notes
•	This implementation does not rely on 3D convolutions.
•	Sequential context is modeled via mesh-cast propagation, allowing any sequence model. Just like in model, we give the lstm, convlstm, transformer, xlstm and mamba
•	The framework is compatible with standard 2D medical segmentation pipelines.


## 📄 Citation

If you find this work useful, please consider citing our paper:

	@inproceedings{Lu2025MNet,
	  title     = {M-Net: MRI Brain Tumor Sequential Segmentation Network via Mesh-Cast},
	  author    = {Lu, Jiacheng and Ding, Hui and Zhang, Shiyu and others},
	  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
	  year      = {2025},
	  pages     = {20116--20125}
	}
	
## 📬 Contact

For questions or discussions, feel free to contact:
	
	Jiacheng Lu
	Capital Normal University
	📧 Email: jchengl@foxmail.com
