
# https://jupyterbook.org/en/stable/start/build.html

# cd pytorch-grad-cam-anim\docs

# jupyter-book build --all mybookname
jupyter-book build .

# https://jupyterbook.org/en/stable/start/publish.html
ghp-import -n -p -f _build/html