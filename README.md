# segmentation-plot

## Description
A Python tool to display high quality image segmentation results as a contour plots using matplotlib functionality.

![Example Segmentation Visualization](/assets/histopath_example.png "Example contour plot of segmentations on a histopathology image.")


## Installation
Install the package from GitHub using pip.
```
pip install git+https://github.com/OnofreyLab/segmentation-plot.git
```



## Usage

```python
from segmentation_plot import segmplot

# Image data in I and segmentation image S (in one-hot encoding)

plt.figure('example', (5, 5))
segmplot.plot_segmentation(
    image=I, 
    segm=[S], 
    segm_cmap=[plt.colormaps['tab10']],
    smooth_sigma=2.0,
    threshold=0.7,
    linewidth=2.0, 
    alpha=1.0, 
    linestyle='-',
)
plt.title("Histopath Example")
# Save as a PDF to utilize the vector graphics of the contour plots.
plt.savefig('histopath_example.pdf', bbox_inches='tight')
plt.show()

```

## Tutorial

See the Jupyter Notebook [tutorial](https://github.com/OnofreyLab/segmentation-plot/blob/main/notebooks/example_histopath.ipynb).



This work was supported by National Institute of Health (NIH) STTR R42 CA224888.
