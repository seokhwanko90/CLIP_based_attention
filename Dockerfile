FROM ubuntu:22.04
RUN apt-get -y update && apt-get install -y --no-install-recommends python3
RUN apt-get -y install openslide-tools

ENV PROJECTDIR=/opt/app
WORKDIR $PROJECTDIR
COPY model $PROJECTDIR
COPY pretrained $PROJECTDIR
COPY pred.py $PROJECTDIR
COPY model_checkpoint.pth $PROJECTDIR

RUN pip3 install numpy
RUN pip3 install torch
RUN pip3 install Pillow
RUN pip3 install timm
RUN pip3 install tifffile
RUN pip3 install openslide-python

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

ENTRYPOINT ["python3", "pred.py"]
