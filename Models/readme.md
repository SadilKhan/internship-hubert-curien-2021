# How to use Ransac-Flow

1. [Download Ransac-Flow](https://github.com/XiSHEN0220/RANSAC-Flow)
3. Replace the ```coarseAlignFeatMatch.py``` in ```~/ransac-flow/quick-start/coarseAlignFeatMatch.py```.
4. Replace the ```outil.py``` in ```~/ransac-flow/utils/outlil.py```.
5. Download a pretrained model.
    * ``` cd ransac-flow ```
    * ``` bash download_model.sh```
6. Or Copy the ```MegaDepth_Theta1_Eta001_Grad0_0.807.pth``` in ```~/ransac-flow/model/pretrained/```. 
7. Run ```ransac.py``` for GUI.

# Functions for every button
1. There are two options to select Source and Target (Reference).
      * ```Choose Folder```: Choose the folder where the images are saved. The last image will be chosen as Reference and rest of them will be chosen as targets.
      * ``` Choose Target ```: Will choose the reference image.
      
        ``` Choose Source ```: Will choose the source images.
 2. Press ``` Start Alignment ``` for alignment matching.
