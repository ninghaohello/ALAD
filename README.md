# ALAD
Accelerated Local Anomaly Detection in Attributed Networks

The whole pipeline is separated into two parts:

(1) Run DSGD_CNMF.py, which implements CNMF and get factor matrices W and H  
(2) Run DSGD_CNMF_FindOutlier.m to detect anomalies (An example result is in the ‘Disney’ folder, in which you can evaluate directly by calling the function)

## Reference in BibTeX:
@inproceedings{liu2017accelerated,
title={Accelerated local anomaly detection via resolving attributed networks},  
author={Liu, Ninghao and Huang, Xiao and Hu, Xia},  
booktitle={Proceedings of the 26th International Joint Conference on Artificial Intelligence},  
pages={2337--2343},  
year={2017},  
organization={AAAI Press}}
