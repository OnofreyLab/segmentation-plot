# segmentation-plot

## Description
A Python tool to display high quality image segmentation results as a contour plots using matplotlib functionality.

![Example Segmentation Visualization](/assets/example.png "Example contour plot of segmentations on synthetic image.")


## Installation
Install the package from GitHub using pip.
```
pip install git+https://github.com/OnofreyLab/segmentation-plot.git
```



## Usage

```python
import segmplot

# Image data in I and segmentation images S

plot_segm = segmplot.PlotSegmentation(
    slice_axis=3, 
    num_slices=5, 
    slice_spacing=3, 
)

plot_segm(I, [S], cmap_name=['tab10'])

```

## Tutorial

See the Jupyter Notebook [tutorial](https://github.com/OnofreyLab/segmentation-plot/blob/main/notebooks/example.ipynb).



This work was supported by National Institute of Health (NIH) STTR R42 CA224888.
