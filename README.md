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
First, please read **Environment** for initial environment setting. Second, For users need to use tools to get images for the specified concept, please go to this Python package[(Link to Github Repo)](https://github.com/hardikvasa/google-images-download) to collect images. Then, for users who want to use their image dataset to obtain the corresponding color probability distribution, please go to **Colorization Network**. And for users who want to map their probability distribution generated from network to get the final color-concept associations(The color library provides the following:  UW-58 colors, UW-71 colors or BCP-37 colors,or you can use the color library you created), please read **Color Mapping** for more help. 
