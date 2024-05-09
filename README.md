1. Git clone the [microSAM](https://github.com/computational-cell-analytics/micro-sam) repo.
2. Install the environment required by the microSAM.
3. Copy the code in [finetuning](finetuning) folder to the example/finetuning folder in the repo.
4. Download [our dataset](https://drive.google.com/file/d/1Q7S9wW9Ksuf3w7p7_w9St_Xa0pHwjc8S/view?usp=drive_link) and extract to the repo.
5. Run finetune_hela.py to train with the baseline mask. Make sure you change directory for the data. 
6. Run train_orig.py to train with the combined mask. Make sure you change directory for the data. You can also download our [model](https://drive.google.com/file/d/1K0pWnQk7Y4nbx1Bhe4VRClGuzh8XZHY9/view?usp=drive_link).
7. Run train_inst.py to train with the instance mask. Make sure you change directory for the data. You can also download our [model](https://drive.google.com/file/d/1hCVCvCNXGxOEg8dkL8jCw-Evz0-06B7C/view?usp=sharing).
8. Run train_axon.py to train with the axon mask. Make sure you change directory for the data. You can also download our [model](https://drive.google.com/file/d/1opKTymw-eH8OQdyWw14vboTE5gzsCQND/view?usp=sharing).
9. Run train_myelin.py to train with the myelin mask. Make sure you change directory for the data. You can also download our [model](https://drive.google.com/file/d/1voie-0zM0iHpO3Nb7kHNo9wE1AlW87na/view?usp=sharing).
10. Run [evaluation.ipynb](evaluation.ipynb) for the example evaluation.
