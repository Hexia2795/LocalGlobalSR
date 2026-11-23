Data

Training uses DF2K: 3,450 HR images from DIV2K and Flickr2K. Evaluation uses Set5, Set14, BSD100, Urban100, Manga109 and the DIV2K validation set.

Sources

DIV2K
Training and validation images; the validation set contains 100 images.
https://data.vision.ee.ethz.ch/cvl/DIV2K/

Flickr2K
Training images. Download instructions are in the Dataset section of the authors' repository.
https://github.com/LimBee/NTIRE2017#dataset

Set5, Set14, BSD100 and Urban100
Benchmark datasets provided with SelfExSR.
https://github.com/jbhuang0604/SelfExSR#datasets

Manga109
https://www.manga109.org/en/

The expected evaluation counts are Set5: 5, Set14: 14, BSD100: 100, Urban100: 100, Manga109: 109 and DIV2K-valid: 100.


Folder layout

Set the data root with:

    python run.py configure --data-root /path/to/data

Evaluation paths are defined in configs/datasets.json. The default benchmark folders are:

    <dataset>/image_SRF_<scale>/HR
    <dataset>/image_SRF_<scale>/LR

For DIV2K validation:

    DIV2K_valid_HR/DIV2K_valid_HR
    DIV2K_valid_LR_bicubic/X<scale>

Training recipes use DF2K_HR and DF2K_LR_X2, DF2K_LR_X3 or DF2K_LR_X4 under the configured data root.

Each LR file must have a matching HR file. The evaluator accepts the same filename, or a scale suffix such as img_001x4.png or img_001X4.png for img_001.png. Image counts and pairs are checked before evaluation.
