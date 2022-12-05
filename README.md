# News-like image captions generator
## Generating News-like Captions for Random Images
### Sugam Arora, Mark Liao, Wesley Liu, Yufei Shen, Jason Wang, Sofie Kardonik
### The University of Texas at Austin
Blog post: https://medium.com/@liaozhoudi/generating-news-captions-for-random-images-e8519aeaa34c \
Github restricts sizes for uploaded files so no training data or trained models are included in this repo. To download the complete repo with training data and trained models go to https://utexas.box.com/s/bl2u7k4vfjaddvl3sycgo3xpcmqz7x8o
## Get Started
1. To run on GPU, install CUDA and cuDNN. (Tutorial: https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781). 
2. Additionally, copy cudnn_ops_infer64_8.dll, cudnn_cnn_infer64_8.dll, cudnn_adv_infer64_8.dll from cudnn_path/bin/ to cuda_path/bin/. 
3. Download ZLIB_DLL from https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-zlib-windows and copy zlibwapi.dll from extracted files to cuda_path/bin.
