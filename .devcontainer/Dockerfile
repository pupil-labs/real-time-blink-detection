FROM ubuntu:22.04

SHELL ["bash","-c"]

ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG USERNAME=vscode

# Create the user
RUN groupadd --gid $USER_GID $USERNAME
RUN useradd --uid $USER_UID --gid $USER_GID -g video -m $USERNAME

# [Optional] Add sudo support. Omit if you don't need to install software after connecting.
RUN apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        python3-dev \
        python3-pip \
        python3-venv \
        python3-opengl \
        xorg-dev \
        cmake \
        git \
        curl \
        wget \
        libglib2.0-0 \
        libgl1-mesa-dri \
        libgl1-mesa-glx \
        libosmesa6 \
        libgl1-mesa-glx \
        libsm6 \
        libxext6 \
        mesa-utils \
        xvfb \
        aptitude \
        ubuntu-standard \
        ubuntu-desktop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY ../requirements.txt .devcontainer/noop.txt /tmp/python-requirements/
RUN if [ -f "/tmp/python-requirements/requirements.txt" ]; then pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/python-requirements/requirements.txt ; fi

USER $USERNAME
CMD ["/bin/bash"]

ENV SHELL /bin/bash
