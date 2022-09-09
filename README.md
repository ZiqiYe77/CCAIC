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
First, please read **Environment** for initial environment setting. Then, for users who want to just use our pre-trained model, please go to **Colorization Network**. And for users who want to map their probability distribution generated from network to the final color-concept associations, please read **Color Mapping** for more help. 
