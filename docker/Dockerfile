
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

LABEL maintainer="Dmitrii Bychkov <dmitrii.bychkov@helsinki.fi>"

ARG PYTHON_VERSION=3.7
ARG CONDA_DIR=/opt/conda
ARG USERNAME=trooper
ARG USER_ID

# Basic packages:
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    build-essential \
    checkinstall \
    libreadline-gplv2-dev \
    libncursesw5-dev \
    libssl-dev \
    libgdbm-dev \
    libc6-dev \
    libbz2-dev \
    openssl \
    libffi-dev \
    pkg-config \
    zlib1g-dev \
    libglib2.0-dev \
    libcairo2-dev \
    libgdk-pixbuf2.0-dev \
    libxml2-dev \
    automake \
    cmake \
    git \
    curl \
    unzip \
    wget \
    unzip \
    htop \
    tmux \
    nano \
    texlive-latex-base \
    texlive-latex-extra \
    ca-certificates \
    libsqlite3-dev \
    sqlite3 \
    libjpeg-dev \
    libpng-dev \
    libtiff5 \
    libtiff-dev \
    libgtk2.0-0 \
    libcanberra-gtk-module \
    libsm6 \
    libxtst6 \
    xvfb \
    libxss1 \
    libnss3-dev \
    libgconf2-4 \
    imagemagick \
    && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it:
RUN useradd -m $USERNAME && echo "$USERNAME:$USERNAME" | chpasswd && adduser $USERNAME sudo
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

ENV HOME=/home/$USERNAME
RUN chmod 777 /home/$USERNAME

RUN wget https://github.com/plotly/orca/releases/download/v1.1.1/orca-1.1.1-x86_64.AppImage -P /home
RUN chmod 777 /home/orca-1.1.1-x86_64.AppImage

# To avoid the need for FUSE, extract the AppImage into a directory (name squashfs-root by default)
RUN cd /home && /home/orca-1.1.1-x86_64.AppImage --appimage-extract
RUN printf '#!/bin/bash \nxvfb-run --auto-servernum --server-args "-screen 0 640x480x24" /home/squashfs-root/app/orca "$@"' > /usr/bin/orca
RUN chmod 777 /usr/bin/orca
RUN chmod -R 777 /home/squashfs-root/

# Install Miniconda
RUN curl -o ~/miniconda.sh -sSL  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x ~/miniconda.sh 
RUN ~/miniconda.sh -bfp /opt/conda 
RUN rm ~/miniconda.sh

RUN ls /opt/conda/

RUN  /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython jupyterlab mkl mkl-include cython typing && \
     /opt/conda/bin/conda install  -y -c menpo opencv && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
     /opt/conda/bin/conda config --add channels conda-forge && \
     /opt/conda/bin/conda install -y -c conda-forge python-spams && \
     /opt/conda/bin/conda clean -ya

RUN chown $USERNAME $CONDA_DIR -R
ENV PATH /opt/conda/bin:$PATH

USER $USERNAME

RUN pip install --user --upgrade pip

ADD requirements.txt .
RUN pip install -r requirements.txt

# Set-up File System:
RUN mkdir /home/$USERNAME/src/
RUN mkdir /home/$USERNAME/data/

WORKDIR /home/$USERNAME/

# TMUX setup:
COPY docker/.tmux.conf  .
COPY docker/tmux-setup.sh .

# Ports: Jupyter & TensorBoardX
EXPOSE 8899
EXPOSE 8097

# Volumes:
VOLUME /home/$USERNAME/src/
VOLUME /home/$USERNAME/data/

CMD ["./tmux-setup.sh"]
ENV SHELL=/bin/bash
