
# Faster R-CNN Research and Implementation

As the title, this repo is for Faster R-CNN implementation.

# Installation:
1. Install Anaconda if you have not already.
2. Create conda environment from file:
  ```
  conda env create -f environment.yml
  ```
3. Activate the conda environment:
  ```
  conda activate rcnn
  ```
4. Go to [Google's open image dataset](https://storage.googleapis.com/openimages/web/download.html) and download the [**classname description**](https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv) file and the [**box annotation**](https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv) files of the subset and put them in the dataset folder.
5. Config the right path in [`/dataset/data_selector.py`](./dataset/data_selector.py) to generate 2 files and create 1 folder:
  * a file containing image id to be downloaded (`data.txt` in this case).
  * a file which is the small version of the annotation file (`train-annotations-bbox.csv` in this case).
  * a folder containing the images to be downloaded (`open_image` in this case).
6. Run the [`/dataset/downloader.py`](./dataset/downloader.py) to download image into a folder.
```
# Cahnge directory to dataset folder.
cd dataset
# For the case we want to use 2 processor.
python downloader.py data.txt --download_folder=open_image --num_processes=2
```
7. Start using the `notebook.py` file.

# Project struture:
* [`notebook.py`](./notebook.py): 
* [`environment.yml`](./environment.yml): 
* `dataset` module:
  * [`dataset/downloader.py`](./dataset/downloader.py): 
  * [`dataset/data_selector.py`](./dataset/data_selector.py): 
* `frcnn` module:
  * [`frcnn/__init__.py`](./frcnn/__init__.py): 
  * [`frcnn/data.py`](./frcnn/data.py): 
  * [`frcnn/losses.py`](./frcnn/losses.py): 
  * [`frcnn/models.py`](./frcnn/models.py): 
  * [`frcnn/utils.py`](./frcnn/utils.py): 


# Project Backlog

<details>
  <summary> <h1>Other works</h1> </summary>
  
  ## Week 1: Feb 1 - Feb 5: [Chest X-ray project week 1](https://github.com/rxng8/Chest-Xray-Abnormalities-Detection#week-1-feb-1---feb-5)

  ## Week 2: Feb 8 - Feb 12: [Chest X-ray project week 2](https://github.com/rxng8/Chest-Xray-Abnormalities-Detection#week-2-feb-8---feb-12)

  ## Week 3: Feb 15 - Feb 19: [Chest X-ray project week 3](https://github.com/rxng8/Chest-Xray-Abnormalities-Detection#week-3-feb-15---feb-19)

  ## Week 4: Feb 22 - Feb 26: [Chest X-ray project week 4](https://github.com/rxng8/Chest-Xray-Abnormalities-Detection#week-4-feb-22---feb-26)

  ## Week 5: March 1 - March 5: [Chest X-ray project week 5](https://github.com/rxng8/Chest-Xray-Abnormalities-Detection#week-5-march-1---march-5)

  ## Week 6: March 8 - March 12: [Chest X-ray project week 6](https://github.com/rxng8/Chest-Xray-Abnormalities-Detection#week-6-march-8---march-12)

  ## Week 7: March 15 - March 19: [Chest X-ray project week 7](https://github.com/rxng8/Chest-Xray-Abnormalities-Detection#week-7-march-15---march-19)

  ## Week 8: March 22 - March 26: [Chest X-ray project week 8](https://github.com/rxng8/Chest-Xray-Abnormalities-Detection#week-8-march-22---march-26)

  ## Week 9: March 29 - April 2: [Shopee project week 9](https://github.com/rxng8/ShopeeKaggle#week-9-march-29---april-2)

  ## Week 10: April 5 - April 9: [Shopee project week 10](https://github.com/rxng8/ShopeeKaggle#week-10-april-5---april-9)

  ## Week 11: April 12 - April 15: [Shopee project week 11](https://github.com/rxng8/ShopeeKaggle#week-11-april-12---april-15)

  ## Week 12: April 18 - April 22: [Shopee project week 12](https://github.com/rxng8/ShopeeKaggle#week-12-april-18---april-22)

</details>

## Week 13: April 25 - April 29:
* [**2 hours**] Create the whole project pipeline and template, write documentation in readme.
* [**0.5 hours**] Prepare environment `rcnn` for the project.
* [**4 hours**] Experiment with [Google's open image dataset](https://storage.googleapis.com/openimages/web/download.html) and setup the dataset downloader pipeline (and download).
  * [**0.5 hours**] Read about the process of downloading the data from google api.
  * [**3.5 hours**] Write code and fix bug to generate appropriate files and csv, as well as labels and annotation files.
* [**1.5 hours**] Read the paper [Faster R-CNN](https://arxiv.org/abs/1506.01497) in-depth. Refering to 5 sources of in-depth information about the method.
* [**5.5 hours**] Implement the overall structure of Faster R-CNN:
  * [**0.5 hours**] Build and test transfer learning model with base VGG16 ([`frcnn/models.py`](./frcnn/models.py)).
  * [**2 hours**] Implement `iou` loss, `rpn` classifier and regressor loss, and object classifier loss ([`frcnn/losses.py`](./frcnn/losses.py)).
  * [**3 hours**] Implement the `cal_rpn()` to generate the label for the rpn networks in the data generator.

---------------

# Tentative:

## Week 14: May 2 - May 6:

## Week 15: May 9 - May 13:

## Week 16: May 16 - May 20:

-----------------


# Reference:
1. Javior. 2018. Faster R-CNN: Down the rabbit hole of modern object detection. Trynolabs. [https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/](https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/).
2. Xy, Y. 2018. Faster R-CNN (object detection) implemented by Keras for custom data from Google’s Open Images Dataset V4. Toward Data Science. [https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a](https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a).
3. Weng, L. 2017. Object Detection for Dummies Part 3: R-CNN Family. Lil' Log. [https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html](https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html).
4. Gad, A. F. 2020. Faster R-CNN Explained for Object Detection Tasks. Paperspace Blog. [https://blog.paperspace.com/faster-r-cnn-explained-object-detection/](https://blog.paperspace.com/faster-r-cnn-explained-object-detection/).
5. Geeksforgeek. 2020. Faster R-CNN | ML. Geeksforgeek. [https://www.geeksforgeeks.org/faster-r-cnn-ml/](https://www.geeksforgeeks.org/faster-r-cnn-ml/)
