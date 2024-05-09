1. Download the [files](https://drive.google.com/drive/u/4/folders/1hzald2Zfs5KtF5wr9MtEkgSJGrm2Tjzm) in this folder and place them into this folder
2. Install [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
3. Install [microSAM from mamba](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html)
4. Install [torch-em from mamba](https://github.com/constantinpape/torch-em)
5. Install jupyter on mamba
6. Activate the mamba environment and launch jupyter lab/notebook.
7. Run pip install -r requirements.txt
8. Run train_microsam.ipynb or use best.pt file that we have trained.
9. Copy the code in [finetuning](finetuning) folder to the example/finetuning folder in the repo.
10. Extract our dataset to the repo. Go to the finetuning folder.
11. Run [finetune_hela.py](finetuning/finetune_hela.py) to train with the baseline mask. Make sure you change directory for the data. 
12. Run [train_orig.py](finetuning/train_orig.py) to train with the combined mask. Make sure you change directory for the data. You can also download our [model](https://drive.google.com/file/d/1K0pWnQk7Y4nbx1Bhe4VRClGuzh8XZHY9/view?usp=drive_link).
13. Run [train_inst.py](finetuning/train_inst.py) to train with the instance mask. Make sure you change directory for the data. You can also download our [model](https://drive.google.com/file/d/1hCVCvCNXGxOEg8dkL8jCw-Evz0-06B7C/view?usp=sharing).
14. Run [train_axon.py](finetuning/train_axon.py) to train with the axon mask. Make sure you change directory for the data. You can also download our [model](https://drive.google.com/file/d/1opKTymw-eH8OQdyWw14vboTE5gzsCQND/view?usp=sharing).
15. Run [train_myelin.py](finetuning/train_myelin.py) to train with the myelin mask. Make sure you change directory for the data. You can also download our [model](https://drive.google.com/file/d/1voie-0zM0iHpO3Nb7kHNo9wE1AlW87na/view?usp=sharing).
16. You can train with baseline mask and instance mask using train_microsam.ipynb and train_microsam_with_accurate_prompting.ipynb respectively.
17. Run [evaluation.ipynb](evaluation.ipynb) for the example evaluation.
