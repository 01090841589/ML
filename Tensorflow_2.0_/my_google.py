from os import path
from google.colab import auth
auth.authenticate_user()

from google.colab import drive
drive.mount('/content/gdrive/')

from PIL import Image