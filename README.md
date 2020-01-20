WLASL: A large-scale dataset for Word-Level American Sign Language
============================================================================================

This repository contains the `WLASL` dataset described in "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison".

Please visit the [project homepage](https://dxli94.github.io/WLASL/) for news update.

File Description
-----------------
The repository contains following files:

 * `WLASL_vx.x.json`: JSON file including all the data samples.

 * `data_reader.py`: Sample code for loading the dataset.

 * `video_downloader.py`: Sample code demonstrating how to download data samples.

 * `C-UDA-1.0.pdf`: the Computational Use of Data Agreement (C-UDA) agreement. You must read and agree with the terms before using the dataset.

 * `README.md`: this file.


Data Description
-----------------

* `gloss`: *str*, data file is structured/categorised based on sign gloss, or namely, labels.

* `bbox`: *[int]*, bounding box detected using YOLOv3 of (xmin, ymin, xmax, ymax) convention. Following OpenCV convention, (0, 0) is the up-left corner.

* `fps`: *int*, frame rate (=25) used to decode the video as in the paper.

* `frame_start`: *int*, the starting frame of the gloss in the video (decoding
with FPS=25), *indexed from 1*.

* `frame_end`: *int*, the ending frame of the gloss in the video (decoding with FPS=25). -1 indicates the gloss ends at the last frame of the video.

* `instance_id`: *int*, id of the instance in the same class/gloss.

* `signer_id`: *int*, id of the signer.

* `source`: *str*, a string identifier for the source site.

* `split`: *str*, indicates sample belongs to which subset.

* `url`: *str*, used for video downloading.

* `variation_id`: *int*, id for dialect (indexed from 0).

* `video_id`: *str*, a unique video identifier.

Please be kindly advised that if you decode with different FPS, you may need to recalculate the `frame_start` and `frame_end` to get correct video segments.

Constituting subsets
---------------
As described in the paper, four subsets WLASL100, WLASL300, WLASL1000 and WLASL2000 are constructed by taking the top-K (k=100, 300, 1000 and 2000) glosses from the `WLASL_vx.x.json` file.


License
---------------
Licensed under the Computational Use of Data Agreement (C-UDA). Plaese refer to `C-UDA-1.0.pdf` for more information.

Disclaimer
---------------
All the WLASL data is intended for academic and computational use only. No commercial usage is allowed. We highly respect copyright and privacy. If you find WLASL violates your rights, please contact us.


Citation
--------------

Please cite the WLASL paper if it helps your research:

    @misc{li2019wordlevel,
          title={Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
          author={Dongxu Li and Cristian Rodriguez Opazo and Xin Yu and Hongdong Li},
          year={2019},
          eprint={1910.11006},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
        }


Contacts
------------------
- [Dongxu Li](https://cecs.anu.edu.au/people/dongxu-li): [email](dongxu.li@anu.edu.au)
- [Cristian Rodriguez Opazo](https://www.roboticvision.org/rv_person/cristian-rodriguez/): [email](cristian.rodriguez@anu.edu.au)
- [Xin Yu](https://sites.google.com/site/xinyuanu/): [email](xin.yu@anu.edu.au)
- [Hongdong Li](https://cecs.anu.edu.au/~hongdong): [email](hongdong.li@anu.edu.au)
