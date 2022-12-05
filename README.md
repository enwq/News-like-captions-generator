# News-like image captions generator
## Generating News-like Captions for Random Images
### Sugam Arora, Mark Liao, Wesley Liu, Yufei Shen, Jason Wang, Sofie Kardonik
### The University of Texas at Austin
Blog post: https://medium.com/@liaozhoudi/generating-news-captions-for-random-images-e8519aeaa34c \
Github restricts sizes for uploaded files so no training data or trained models are included in this repo. To download the complete repo with training data and trained models go to https://utexas.box.com/s/bl2u7k4vfjaddvl3sycgo3xpcmqz7x8o.
## Get Started
1. To run on GPU, install CUDA and cuDNN. (Tutorial: https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781). 
2. Additionally, copy cudnn_ops_infer64_8.dll, cudnn_cnn_infer64_8.dll, cudnn_adv_infer64_8.dll from cudnn_path/bin/ to cuda_path/bin/. 
3. Download ZLIB_DLL from https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-zlib-windows and copy zlibwapi.dll from extracted files to cuda_path/bin/.
## Data Preparation
### Get raw data
#### Get Flickr8k data
1. Download data from https://www.kaggle.com/datasets/adityajn105/flickr8k. Put all images under flickrdata/Images and put caption.txt under flickrdata/.
#### Get GoodNews data
1. Go to Github repo https://github.com/furkanbiten/GoodNews.
2. Follow its documentation, download news_dataset.json and put it under newsdata/.
3. Follow its documentation, download article.json from Goodnews' model output and put it under newsdata/.
4. Follow its documentation, download all news images and put them under newsdata/all_images/.
### Generate news captions dataset
1. Run get_news_data.ipynb. It will select 8000 image-caption pairs with named entity recognitions from all GoodNews images .
2. Selected captions are stored in newsdata/news_captions_ner.txt and selected images are in newsdata/Images/.
### Generate named entity candidates
1. Run get_ne_candidates.ipynb. It will classify 10000 random GoodNews images into ImageNet classes and for each image it will extract named entity candidates from its associated article and save them for top 3 predicted classes of the image. 
2. Generated named entity candidates are stored in newsmodels ner_for_classes.json.
