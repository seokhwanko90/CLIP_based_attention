FROM ubuntu:22.04
RUN apt-get -y update && apt-get install -y --no-install-recommends python3

ENV PROJECTDIR=/opt/app
WORKDIR $PROJECTDIR
COPY model $PROJECTDIR
COPY pretrained $PROJECTDIR
COPY pred.py $PROJECTDIR
COPY model_checkpoint.pth $PROJECTDIR

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

ENTRYPOINT ["python3", "pred.py"]
