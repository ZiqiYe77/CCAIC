# Color Mapping

<div align=center><img src="https://github.com/ZiqiYe77/CCAIC/blob/main/docs/ColorMapping.png" width="60%"></div>


(A) <b>Image background pixel processing. </b>
In order to ensure the accuracy of the color-concept association, we need to get rid of pixels that are not relevant to the concept, such as background pixels. Run `tool_masking.py` to implement this step.

(B) <b>Color mapping. </b>
After setting up the specified color library in the `tool_color_mapping.py`, we can obtain the final color concept distribution.

(C) <b>Draw result bars. </b>
Based on the color concept association, we can draw the corresponding bar chart by running `tool_draw_result_bars.py`.

(D) <b>Calculate metrics. </b>
Run `tool_cal_metrics.py`, we can evaluate the results of the color-concept association.
