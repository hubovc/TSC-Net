# TSC-Net-Traj-Pred

Environment Installation
======================

* Create a new conda environment and activate the environment by running
    ```
    conda create -n tscnet python=3.7.11
    conda activate tscnet
    ```
* Install PyTorch by 
   ```
   pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
   ```

* Install other necessary packages by 
    ```
    conda install --file requirements.txt
    ```

* Install the rest packages which cannot be found in conda by 
    ```
    pip install easydict
    pip install pickle5=0.0.12
    ```
    
* Some packages are not used in our model, but they are required when accessing the original data of SDD:
    ```
    pip install segmentation-models-pytorch==0.1.0    
    ```



Data Preparation
======================

The required data includes the trajectory data frame and corresponding scene map images. 

Trajectory Data 
----------------------

Take raw/train.pkl as an example, the data frame looks like

	            frame  trackId      x      y      sceneId  metaId
		0           0      100  757.0  918.5  bookstore_0       0
		...
		19        228      100  830.0  910.5  bookstore_0       0
		20          0      129  897.5  183.0  bookstore_0       1
		...
		39        228      129  744.0  196.5  bookstore_0       1
		...
        ...
        169840  11244      263  411.0  1548.5     nexus_9    8492
		...
		169859  11472      263  601.5  1664.0     nexus_9    8492
		169860  11244      273  256.0   973.0     nexus_9    8493
		...
		169879  11472      273   58.5   932.0     nexus_9    8493

trackId：   pedestrian identity \
metaId:		trajectory index    \
frame: 		frame index of the video \
sceneId: 	scene name 

The data frame is organized as:
* all pedestrians in the same sceneId, e.g., sceneId bookstore_0
    * all trajectories start at the same frame, e.g., frame 0 (the trajectories span in frame 0,12,24,...,228)
        * all pedestrians in this 20-frame clip: e.g., trackId 100
            * all 20 frames for each pedestrian: e.g., metaId 0
                * x and y coordinate in one frame: e.g., 757.0  918.5




Note that: 
* One pedestrian may have multiple different metaId but only have one unique trackId

* In the pedestrian trajectory prediction task, suppose trajectory length is 20 frames, then a 20-frame video clip is treated as a sample. All the pedestrians in this sample should have complete 20-frame trajectory. Incomplete trajectory, such as begin or end in the middle of the clip, is not allowed. 

* In our data pre-processing code, metaId and trackId is not used as all rows are well sorted so that it is easy to extract trajectories from each 20 rows. However, It is suggested that record all information (including metaId and trackId etc.) and organize the data in the same way as train/test.pkl when build a new dataset. 



Scene Data
----------------------
Each sceneId in the trajectory data frame has a corresponding png file. The pixel value of the png file records the semantic label of the location, where the label $\in [0,1,...,5]$. 


Data Source
----------------------
* The original video and annotation data of SDD dataset can be found at [SDD Download](http://vatic2.stanford.edu/stanford_campus_dataset.zip)

* The processed data of SDD dataset can be found at [Pretrained Models, Data and Config files](https://drive.google.com/file/d/1u4hTk_BZGq1929IxMPLCrDzoG3wsZnsa/view?usp=sharing), which is from the repository of [Y-Net](https://github.com/HarshayuGirase/Human-Path-Prediction/tree/master/ynet).


Data Pre-Processing
======================

Run 
```
cd data
python build_split.py
```
for data pre-processing. This step is to prepare the required data during training to accelerate the training process.

The output of data pre-processing includes
* Trajectory coordinates in different types.
* Original scene maps.
* All cells ground truth and mask for loss computing.
* Scene maps centered on goals and each step (it is saved separately as the size is very large, about 240GB for SDD).
* Other information such as start_frame, scene_id, which is helpful for visualization.

To test a new dataset, the following hyper-parameter in config/two_stage_sdd.yaml need to be adjusted:
* **dataset_name**: name of the new dataset
* **ori_global_size**: the center of global grid is centered on the end position of trajectory history. Adjust the ori_global_size to ensure all trajectory end positions in this dataset are covered in the global grid. (See the following figure)
* **global_size**: equal to ori_global_size / resize_factor
* **global_anchor_size**: 15 means there are 15x15 cells in the global grid
* **global_step**: equal to ori_global_size / global_anchor_size
* **ori_local_size**: the center of local grid is centered on the every time step of the whole trajectory. Adjust the ori_local_size to ensure all positions of next step in this dataset are covered in the local grid. 
* **local_size**: equal to ori_local_size / resize_factor
* **local_anchor_size**: 3 means there are 3x3 cells in the global grid
* **local_step**: equal to ori_local_size / local_anchor_size


Note that the ratio between global_size and global_anchor_size is the input and output size of the backbone network. If a new setting is applied, e.g., train on a new dataset, the structure of backbone network need to be modified. The backbone network definition can be found in init function of class Backbone in models/layers.py.

Network Training
======================
Run
```
python main_train.py
```
to train the network from scratch. If the model is trained with an intermediate weight file, the file name should be **weight/TSC_Net_datasetname_Ep_xx**. xx represents the epoch of this weight file obtained. For example, run 
```
python main_train.py 30
```
to train the network with the model initialized by **weight/TSC_Net_SDD_Ep_30**.

Network Testing
======================
Run
```
python main_test.py
```
to test the network for all epochs. When testing the specific weight file, the weight file name should also be **weight/TSC_Net_datasetname_Ep_xx**. xx represents the epoch of this weight file obtained. For example, run 
```
python main_test.py 30
```
to test the network with the weight file **weight/TSC_Net_SDD_Ep_30**.

