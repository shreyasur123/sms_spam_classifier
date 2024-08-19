mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"shreyasur965@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = \$PORT\n\
" > ~/.streamlit/config.toml

# Create an NLTK data directory and download required resources
mkdir -p ~/.nltk_data
python -m nltk.downloader -d ~/.nltk_data wordnet stopwords
