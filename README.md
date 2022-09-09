# Self-Supervised Color-Concept Association via Image Colorization
![image](https://github.com/ZiqiYe77/CCAIC/blob/main/docs/teaser.png)
**This is the author's code release for:**
> **Self-Supervised Color-Concept Association via Image Colorization**  
> Ruizhen Hu, Ziqi Ye, Bin Chen, Oliver van Kaick, Hui Huang. <br>
> **IEEE Transactions on Visualization and Computer Graphics (Proceedings of InfoVis 2022), 2022.**

##  Introduction
We introduce a self-supervised method for automatically extracting color-concept associations from natural images. We apply a colorization neural network to predict color distributions for input images. The distributions are transformed into ratings for a color library, which are then aggregated across multiple images of a concept.


![image](https://github.com/ZiqiYe77/CCAIC/blob/main/docs/overview.png)

## Getting started
1. Please read **Environment** for initial environment setting. 
2. For users need to collect images for the specified concept, please go to **Data Collection** to get Google Images for download. 
3. For users who want to use their image dataset to obtain the corresponding color probability distribution, please go to **Colorization Network**. 
4. For users who want to map their probability distribution generated from network to get the final color-concept associations(The color library provides the following:  UW-58 colors, UW-71 colors or BCP-37 colors, or you can use the color library you created), please read **Color Mapping** for more help. 

* [Environment](https://github.com/ZiqiYe77/CCAIC/blob/main/Environment/README.md)
* [Data Collection](https://github.com/hardikvasa/google-images-download)
* [Colorization Network](https://github.com/ZiqiYe77/CCAIC/tree/main/Colorization%20Network)
* [Color Mapping](https://github.com/ZiqiYe77/CCAIC/tree/main/Color%20Mapping)
