FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Set a specific user, UID, and GID
ARG USERNAME=debsen
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Update and install debian stuff
RUN apt-get update && apt-get -y install \
    wget \
    tmux \
    unzip \
    git \
    curl \
    aptitude \
    vim \
    tree \
    software-properties-common \
    lsb-release \
    manpages-dev \
    build-essential \
    libgl1-mesa-glx \
    mesa-utils\
    libboost-dev \
    libxerces-c-dev \
    libeigen3-dev\
    python-is-python3 \
    python3-pip \
    python3-tk \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# python and pytorch specific version
RUN pip install --upgrade pip \
    pip install --upgrade wheel \
    pip install  https://github.com/isl-org/open3d_downloads/releases/download/torch1.8.2/torch-1.8.2-cp38-cp38-linux_x86_64.whl

COPY requirements.txt requirements.txt

RUN python -m pip install -r requirements.txt

EXPOSE 8888

# install GDAL/OGR
RUN add-apt-repository ppa:ubuntugis/ppa

RUN aptitude -y install \
    gdal-bin \
    libgdal-dev

# Install sudo and create the user
RUN apt-get update && apt-get -y install sudo && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to the new user
USER $USERNAME

RUN echo "bind r source-file ~/.tmux.conf \; display \"Reloaded!\"" >> ~/.tmux.conf && \
    echo "set -g mouse on" >> ~/.tmux.conf && \
    echo "set -s escape-time 0" >> ~/.tmux.conf && \
    echo "set-option -g default-shell /bin/bash" >> ~/.tmux.conf && \
    echo "unbind C-Space" >> ~/.tmux.conf && \
    echo "set -g prefix C-Space" >> ~/.tmux.conf && \
    echo "bind C-Space send-prefix" >> ~/.tmux.conf && \
    echo "set-option -g history-limit 5000" >> ~/.tmux.conf && \
    echo "set -g base-index 1" >> ~/.tmux.conf && \
    echo "setw -g pane-base-index 1" >> ~/.tmux.conf && \
    echo "set -g renumber-windows on" >> ~/.tmux.conf && \
    echo "bind | split-window -hc \"#{pane_current_path}\"" >> ~/.tmux.conf && \
    echo "bind - split-window -vc \"#{pane_current_path}\"" >> ~/.tmux.conf && \
    echo "setw -g mode-keys vi" >> ~/.tmux.conf && \
    echo "# List of plugins" >> ~/.tmux.conf && \
    echo "set -g @plugin 'tmux-plugins/tpm'" >> ~/.tmux.conf && \
    echo "set -g @plugin 'tmux-plugins/tmux-sensible'" >> ~/.tmux.conf && \
    echo "set -g @plugin 'dracula/tmux'" >> ~/.tmux.conf && \
    echo "set -g @plugin 'tmux-plugins/tmux-yank'" >> ~/.tmux.conf && \
    echo "# Initialize TMUX plugin manager (keep this line at the very bottom of tmux.conf)" >> ~/.tmux.conf && \
    echo "run '~/.tmux/plugins/tpm/tpm'" >> ~/.tmux.conf 

RUN git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm

RUN sudo apt-get install xsel && \
    sudo apt-get -y install xclip 

#RUN pip install git+https://github.com/autonomousvision/kitti360Scripts.git && \
#    pip install Pillow==6.1
    