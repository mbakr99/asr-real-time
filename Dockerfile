FROM ubuntu:20.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.7 and required dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    # python3.7-distutils \
    build-essential \
    gcc \
    g++ \
    wget \
    cmake \
    git

# Install pip for Python 3.7
# RUN wget https://bootstrap.pypa.io/get-pip.py && python3.7 get-pip.py && rm get-pip.py

# Install NumPy for Python 3.7
#RUN pip3 install numpy

# Set the working directory inside the container
WORKDIR /workspace

# Copy the source files into the container
COPY . /workspace

# Command to keep the container running for testing
CMD ["/bin/bash"]
