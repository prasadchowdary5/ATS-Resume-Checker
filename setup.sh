#!/bin/bash
mkdir -p /home/adminuser/nltk_data
python -c "import nltk; nltk.download('punkt', download_dir='/home/adminuser/nltk_data'); nltk.download('punkt_tab', download_dir='/home/adminuser/nltk_data'); nltk.download('stopwords', download_dir='/home/adminuser/nltk_data')"