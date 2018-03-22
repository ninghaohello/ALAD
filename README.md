# ALAD
Accelerated local anomaly detection via resolving attributed networks

The whole pipeline is separated into two parts:  
(1) Run DSGD_CNMF.py in python, which implements CNMF and get factor matrices W and H  
(2) Run DSGD_CNMF_FindOutlier.m in MATLAB to detect anomalies  
(An example result is in the ‘disney’ folder, in which you can evaluate directly by calling DSGD_CNMF_FindOutlier.m)

## Reference in BibTeX:
@inproceedings{liu2017accelerated,  
title={Accelerated local anomaly detection via resolving attributed networks},  
author={Liu, Ninghao and Huang, Xiao and Hu, Xia},  
booktitle={Proceedings of the 26th International Joint Conference on Artificial Intelligence},  
pages={2337--2343},  
year={2017},  
organization={AAAI Press}}
