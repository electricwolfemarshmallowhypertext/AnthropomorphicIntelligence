SAVE_DIR=~/HumanLLM_data/raw
mkdir -p ${SAVE_DIR}

wget -c "https://huggingface.co/datasets/barilan/blog_authorship_corpus/resolve/main/data/blogs.zip" -P ${SAVE_DIR}


python blogger_data_filter.py