FROM ubuntu:20.04

# Install python3
RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install python3-pip

# Rename python3 command to python, pip3 to pipi
# RUN ln -s /usr/bin/python3 /usr/bin/python
# RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Copy requirements.txt to working directory
COPY scripts/reqs.txt reqs.txt
# Install Python packages defined in reqs.txt
RUN apt-get update

RUN pip install -r reqs.txt

# Start the container
ENTRYPOINT  ["tail", "-f", "/dev/null"]
