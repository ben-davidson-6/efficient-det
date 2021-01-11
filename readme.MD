# EfficientDet in Tensorflow 2.3: work in progress

Object detection from scratch in tensorflow 2.4. Below are the best results so far for the WiderFace dataset.


```
Average Precision  (AP) @[ IoU=0.50:0.50 | area=   all | maxDets=100 ] = 0.314
Average Precision  (AP) @[ IoU=0.50:0.50 | area= small | maxDets=100 ] = 0.223
Average Precision  (AP) @[ IoU=0.50:0.50 | area=medium | maxDets=100 ] = 0.835
Average Precision  (AP) @[ IoU=0.50:0.50 | area= large | maxDets=100 ] = 0.926
Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=100 ] = 0.400
Average Recall     (AR) @[ IoU=0.50:0.50 | area= small | maxDets=100 ] = 0.319
Average Recall     (AR) @[ IoU=0.50:0.50 | area=medium | maxDets=100 ] = 0.907
Average Recall     (AR) @[ IoU=0.50:0.50 | area= large | maxDets=100 ] = 0.965
```

<div align="center">
<img src="images/val.png" alt="validation image example" style="display: block; margin-left: auto; margin-right: auto; width: 80%;" border="10" />
</div>
