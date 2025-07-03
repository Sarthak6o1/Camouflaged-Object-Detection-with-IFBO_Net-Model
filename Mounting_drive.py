import torch

print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("CUDA not available. Running on CPU.")

from google.colab import drive
drive.mount('/content/drive')

from google.colab import drive
drive.mount('/content/drive')
!unzip "/content/drive/MyDrive/archive.zip" -d /content/

import os
print(os.listdir('/content/COD10K-v3'))
