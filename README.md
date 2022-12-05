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
2. Generated named entity candidates are stored in newsmodels/ner_for_classes.json.
## Training
### Train models with only news captions dataset
1. Run GoodNews.ipynb. It will train an image captioner on the 8000 news image-caption pairs generated before.
2. Resulting train-test split image ids are stored in newsmodels/news_test_id.txt and newsmodels/news_train_id.txt respectively.
3. Resulting image features extracted from VGG16 are stored in newsmodels/newsfeaturesextracted.pkl.
4. Resulting tokenizer is stored in newsmodels/tokenizer.pkl.
5. Resulting models trained for 10 epochs and 20 epochs are stored in newsmodels/newstenepochs.h5 and newsmodels/newstwentyepochs.h5 respectively.
### Train models with Flick8k images then fine-tune on news captions dataset
#### Use tokenizer that distinguishes between named entities and normal words (e.g. named entity placeholder \<person\> and word person will be treated as different tokens)
1. Run finetuning.ipynb. It will train an image captioner on Flickr8k images first then fine-tune the trained model on news captions dataset.
3. Resulting image features for Flickr8k images extracted from VGG16 are stored in finetuning/flickrfeaturesextracted.pkl.
4. Resulting tokenizer is stored in finetuning/tokenizer.pkl
5. Resulting base model trained on Flickr8k images for 10 epochs are stored in finetuning/finetuningbase.h5.
6. Resulting models fine-tuned on news captions dataset for 5 epochs and 15 epochs are stored in finetuning/finetuningfive.h5 and finetuning/finetuningfifteen.h5 respectively.
#### Use tokenizer that does not distinguish between named entities and normal words
1. Run finetuning.ipynb. It will train an image captioner on Flickr8k images first then fine-tune the trained model on news captions dataset.
3. Resulting image features for Flickr8k images extracted from VGG16 are stored in finetuning/flickrfeaturesextracted.pkl.
4. Resulting tokenizer is stored in finetuning/tokenizer_mix.pkl
5. Resulting base model trained on Flickr8k images for 10 epochs are stored in finetuning/finetuningbase_mix.h5.
6. Resulting models fine-tuned on news captions dataset for 5 epochs and 15 epochs are stored in finetuning/finetuningfive_mix.h5 and finetuning/finetuningfifteen_mix.h5 respectively.
## Demo
1. Run news_caption_generation.ipynb. It will generate news-like captions for an input image url with the trained newstwentyepochs.h5 and finetuningfifteen_mix.h5 models.
