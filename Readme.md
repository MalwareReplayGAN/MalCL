## MalCL: Leveraging GAN-Based Generative Replay to Combat Catastrophic Forgetting in Malware Classification

---

This repository contains the codes of the paper __MalCL: Leveraging GAN-Based Generative Replay to Combat Catastrophic Forgetting in Malware Classification__



### Dataset
---
The dataset for the experiments can be downloaded from [here](https://drive.google.com/drive/folders/1YGmxcQGqu22ZQuZccpD81WUBKHh7c3Jq?usp=sharing)

* EMBER 2018 dataset    
We use the 2018 EMBER dataset, known for its challenging classification tasks, focusing on a subset of 337,035 malicious Windows PE files labeled by the top 100
malware families, each with over 400 samples. Features include file size, PE and COFF header details, DLL characteristics, imported and exported functions, and properties
like size and entropy, all computed using the feature hashing trick.
* AZ-Class    
The AZ-Class dataset contains 285,582 samples from 100 Android malware families, each with at least 200 samples. We extracted Drebin features (Arp et al.2014) from the apps, covering eight categories like hardware access, permissions, API calls, and network addresses.

### Environment
---
* pytorch version 2.0.1
* conda version 4.7.12
* python version 3.8.13
* NVIDIA RTX A6000
* CUDA version 11.4


### Pipeline
---
![pipeline](https://github.com/MalwareReplayGAN/MalCL/blob/master/Repo_img/pipeline_new.png)


### Architecture
---
* Generator


![Generator](https://github.com/MalwareReplayGAN/MalCL/blob/master/Repo_img/Generator.png)


* Discriminator

  
![Discriminator](https://github.com/MalwareReplayGAN/MalCL/blob/master/Repo_img/Discriminator.png)


* Classifier

  
![Classifier](https://github.com/MalwareReplayGAN/MalCL/blob/master/Repo_img/Classifier.png)


### Results
---

![Table](https://github.com/MalwareReplayGAN/MalCL/blob/master/Repo_img/table.png)    
* Comparisons to Baseline and Prior Replay Models Using Ember Dataset. We report the mean accuracy scores (Mean) and minimum (Min) computed from every 11 tasks.    

![graph](https://github.com/MalwareReplayGAN/MalCL/blob/master/Repo_img/EMBERvsAZ.png)    
* MalCL Performance on the EMBER and AZ-Class datasets using FML and L1-norm to Mean Logits.    



