
# Matching Networks Tensorflow V2 Implementation

## Introduction
This repository contains an implementation of [Matching Networks](https://arxiv.org/abs/1606.04080) in Tensorflow 2 (currently in BETA).
It is specifically designed for few-shot learning of 1D data sets (e.g. flattend MNIST).
However, I've added various tools for users so that they can make minor customizations (e.g. 2D/3D Convolution).

## Usage
```bash
python main.py --train/test --seed=X --C=Y --K=Z
```

In the config.py file, you should change ```project_path```, ```result_path```, ```data_path``` to fit your needs.

Or you may look into ```MNforADNI.ipynb``` or [Colab](https://colab.research.google.com/github/OpenXAIProject/One-Shot-Learning/blob/master/MNforADNI.ipynb) for a demo.
 
# XAI Project 

**These works were supported by Institute for Information & Communications Technology Promotion (IITP) grant funded by the Korea government (MSIT) (No.2017-0-01779, A machine learning and statistical inference framework for explainable artificial intelligence)**

+ Project Name : A machine learning and statistical inference framework for explainable artificial intelligence(의사결정 이유를 설명할 수 있는 인간 수준의 학습·추론 프레임워크 개발)

+ Managed by Ministry of Science and ICT/XAIC <img align="right" src="http://xai.unist.ac.kr/static/img/logos/XAIC_logo.png" width=300px>

+ Participated Affiliation : UNIST, Korea Univ., Yonsei Univ., KAIST, AItrics  

+ Web Site : <http://openXai.org>


### License
[Apache License 2.0](https://github.com/OpenXAIProject/tutorials/blob/master/LICENSE "Apache")

### Contact
Yoon, Jee Seok  
[wltjr1007@korea.ac.kr](emailto:wltjr1007@korea.ac.kr)  
Graduate Student,  
[Machine Intelligence Lab.](https://milab.korea.ac.kr),  
Department of Brain and Cognitive Engineering,  
Korea University, Seoul, South Korea
