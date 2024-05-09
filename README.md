1. Git clone the [microSAM](https://github.com/computational-cell-analytics/micro-sam) repo.
2. Install the environment required by the microSAM.
3. Copy the code in finetuning folder to the example/finetuning folder in the repo.
4. Download [our dataset](https://drive.google.com/file/d/1Q7S9wW9Ksuf3w7p7_w9St_Xa0pHwjc8S/view?usp=drive_link) to the repo.
5. Run finetune_hela.py to train with the baseline mask.
6. Run train_orig.py to train with the combined mask.
7. Run train_inst.py to train with the instance mask.
8. Run train_axon.py to train with the axon mask.
9. Run train_myelin.py to train with the myelin mask.
10. Run evaluation.ipynb for the example evaluation.
