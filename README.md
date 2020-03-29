WLASL: A large-scale dataset for Word-Level American Sign Language
============================================================================================

This repository contains the `WLASL` dataset described in "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison".

Please visit the [project homepage](https://dxli94.github.io/WLASL/) for news update.

Download
-----------------
1. Download repo.
```
git clone https://github.com/dxli94/WLASL.git
```

2. Install [youtube-dl](https://github.com/ytdl-org/youtube-dl#installation) for downloading YouTube videos.
3. Download raw videos.
```
cd start_kit
python video_downloader.py
```
4. Extract video samples from raw videos.
```
python preprocess.py
```

Requesting Missing Videos
-----------------

1.  Videos can dissapear over time due to expired urls, so you may find the videos you downloaded incomplete, we provide the following solution for you to have access to missing videos.

 (a) Run
```
python find_missing.py
```
to generate text file missing.txt containing missing video IDs.

 (b)  Submit a video request by agreeing to terms of use at:  https://docs.google.com/forms/d/e/1FAIpQLSc3yHyAranhpkC9ur_Z-Gu5gS5M0WnKtHV07Vo6eL6nZHzruw/viewform?usp=sf_link. You will get links to the missing videos within 72 hours.
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


FAQ
---------------
**File formats**

Q1. Do you convert .swf files? / Do you convert everything to .mp4?

A1. Generally depends on how you feed the videos into the model. But yes, in our implementations, we use ffmpeg to convert all files into .mp4 format to a unified data io.

**Connection Error**

Q2. Connection forcibly closed by remote server?

A2. First, manually access the URL and ensure it is valid or not. If it is invalid, please report to us via email and we will look into it. Otherwise, it is likely you are requesting too frequently. Try adding pauses between your requests to avoid the issue.

**Missing Videos**

Q3. I encountered 404 error. / Downloader is not able to download certain videos.

A3. If you see a lot of broken URLs, please see Q2. If you have a dozens of broken URLs, please first manually check whether they are valid in your browser. Then you can either choose to manually save the videos or report to us for invalid cases. If you have only a few of videos missing because of deprecated URLs or connections, you may request for the missing ones by email to dongxu.li@anu.edu.au.


TODO
--------------
1. Adding a preprocess script.
2. Release training models.


License
---------------
Licensed under the Computational Use of Data Agreement (C-UDA). Plaese refer to `C-UDA-1.0.pdf` for more information.

Disclaimer
---------------
All the WLASL data is intended for academic and computational use only. No commercial usage is allowed. We highly respect copyright and privacy. If you find WLASL violates your rights, please contact us.


Citation
--------------

Please cite the WLASL paper if it helps your research:

     @inproceedings{li2020word,
        title={Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
        author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
        booktitle={The IEEE Winter Conference on Applications of Computer Vision},
        pages={1459--1469},
        year={2020}
     }

Please consider citing our work on WLASL.

    @article{li2020transferring,
      title={Transferring Cross-domain Knowledge for Video Sign Language Recognition},
      author={Li, Dongxu and Yu, Xin and Xu, Chenchen and Petersson, Lars and Li, Hongdong},
      journal={arXiv preprint arXiv:2003.03703},
      year={2020}
    }


Revision History
--------------
* WLASLv0.3 (Mar. 16, 2020): updated dead URL links. Added a script for downloading non-YouTube videos.
* WLASLv0.2 (Mar. 11, 2020): updated URL links for ASL signbank.


Contacts
------------------
- [Dongxu Li](https://cecs.anu.edu.au/people/dongxu-li): [email](dongxu.li@anu.edu.au)
- [Hongdong Li](https://cecs.anu.edu.au/~hongdong): [email](hongdong.li@anu.edu.au)

Please send queries with your institude mail address.
