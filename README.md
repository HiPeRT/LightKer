# Light Kernel

<img src="img/lk-cuda-logo.png" style="height:100px">

## What it is
LightKer is a free implementation of the persistent thread paradigm for  General Purpose GPU systems. It acts as a flexible layer to "open the box" of a GPU and enables 
direct allocation of work to specific cluster(s) of GPU cores.

## What is the status?
Currently, we released the v0.2 of LightKer. It is perfectly working, please see the <em>example1</em> excerpt of code which you might want to use as a starting point to develop your LK-compliant application.
Please note that the current version <i>is a work-in-progress</i>, which we release to ease discussion and joint work on it.

## How can I get it?
LK is distributed under <a href="https://www.gnu.org/licenses/gpl.html" target="_blank">GPL v3 license</a> + <a href="https://en.wikipedia.org/wiki/GPL_linking_exception">linking exception</a> .
Please download preliminary version 0.2 <a href="https://github.com/HiPeRT/LightKer" target="_blank">here</a>.

## Using and citing LK

We provide LK as free software, under GPL license with linking exception to make it usable also by proprietary SW developers.

This project accompanies the following publication. Please, cite/acknowledge this if you plan to use the code.  - 
[Link to paper](https://arxiv.org/abs/2310.01212)

```
@misc{burgio2023enabling,
      title={Enabling predictable parallelism in single-GPU systems with persistent CUDA threads}, 
      author={Paolo Burgio},
      year={2023},
      eprint={2310.01212},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```

# Scheduled Light Kernel

<img src="img/lk-ga-logo.png" style="height:20px">

Scheduled Light Kernel (SLK) is a version of LK used to accelerate  Genetic Algorithms. Please refer to the following paper for more information:

```
@INPROCEEDINGS {7387293,
author = {N. Capodieci and P. Burgio},
booktitle = {2015 Seventh International Symposium on Parallel Architectures, Algorithms and Programming (PAAP)},
title = {Efficient Implementation of Genetic Algorithms on GP-GPU with Scheduled Persistent CUDA Threads},
year = {2015},
volume = {},
issn = {2168-3042},
pages = {6-12},
doi = {10.1109/PAAP.2015.13},
url = {https://doi.ieeecomputersociety.org/10.1109/PAAP.2015.13},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {dec}
}
```

# Acknowledgements
LK is a side research project of the <a href="http://hercules2020.eu/" target="_blank">Hercules Project</a> from the European Union's Horizon 2020 research and innovation programme under grant agreement No. 688860. 
<br>
Special thanks to [Serena](https://github.com/sere), who implemented the first version of LK

# Authors
* **Serena Ziviani** - [sere](https://github.com/sere)
* **Arianna Avanzini** - [ariava]*https://github.com/ariava)
* **Paolo Burgio** - [pburgio](https://github.com/pburgio)
* **Nicola Capodieci** - [ncapodieci](https://git.hipert.unimore.it/ncapodieci)
