# DLC: Deep Learning Content Caching
### A pytorch GPU implementation our accepted paper "End-to-End Deep Learning Proactive Content Caching Framework" at Globecom 2022 CSM

## Authors:
    * Eslam Mohamed BAKR
    * Hamza Ben-Ammar '
    * Hesham Eraqi '
    * Sherif G. Aly
    * Tamer Elbatt
    * Yacine Ghamri-Doudane

' = Equal contribution

## Abstract:
Proactive content caching has been proposed as a
promising solution to cope with the challenges caused by the
rapid surge in content access using wireless and mobile devices
and to prevent significant revenue loss for content providers. In
this paper, we propose an end-to-end Deep Learning framework
for proactive content caching that models the dynamic interaction
between users and content items, particularly their features.
The proposed model performs the caching task by building a
probability distribution across different content items, per user,
via a Deep Neural Network model and supports, both, centralized
and distributed caching schemes. In addition, the paper addresses
the key question: Do we need an explicit user-item pairs-based
recommendation system in content caching? i.e., do we need to
develop a recommendation system while tackling the content
caching problem? To this end, an end-to-end Deep Learning
framework is introduced. Finally, we validate our approach
through extensive experiments on a real-world, public data set,
coined MovieLens. Our experiments show consistent performance
gains against its counterparts, where our proposed Deep Learning
Caching module, dubbed as DLC, significantly outperforms state-
of-the-art content caching schemes, serving as a baseline.

## The requirements are as follows:
	* python==3.7
	* pandas==0.24.2
	* numpy==1.16.2
	* pytorch==1.7.1
	* gensim==3.7.1
	* tensorboardX>=1.6 (mainly useful when you want to visulize the loss, see https://github.com/lanpa/tensorboard-pytorch)
Please note that, we have provided a pre-made conda environment, named content_caching", where you can directly install it via the following command:

    conda env export > content_caching.yml

## Example to run:
Download the data from this [link](https://drive.google.com/drive/folders/1Keww2JHH4Pqx_Oe5Q6hp641t-weU2vPd?usp=sharing)
and put the "data" folder under DLC folder.

## Example to run:
```
python main.py --batch_size=256 --lr=0.001 --factor_num=16
```
### Configurations:
All the needed configurations are listed in config.py:

    * time_sorting --> If activated we will sort the data based on the time-stamp, otherwise will be distributed to make sure that all the users are represented in each time-interval.
    * rating_th --> Filter data based on sorting. for example remove items that got rating less than 3 to consider positive rating starting from 4 only. If set to zero that means no filtration will occur.
    * window_split --> If set, the data will be split into windows. Otherwise if zero the all data will be used. Please note that this number in thousands.
    * E2E --> E2E: If activated that means we will train the neural network model to directly predict top-k. one stage instead of two stages.
    * modes --> choose whether "training" or "testing".
    * user_item_info --> Use extra info for the user and items while training and testing.

## Benchmarking against other content caching algorithms:
![Alt text](CFR_benchmarking.png?raw=true "Title")

We used the following repos to run the other content caching algorithms, i.e., FIFO, GDS, FOO , etc.

    * https://github.com/dasebe/webcachesim
    * https://github.com/dasebe/optimalwebcaching

## Please cite our work:
Eslam Bakr, Hamza B. Ammar, Hesham M. Eraqi, Sherif Aly, Yacine Ghamri-Doudane, Tamer Elbatt. End-to-End Deep Learning Proactive Content Caching Framework. IEEE Global Communications Conference. Rio de Janeiro, Brazil, December 2022.
