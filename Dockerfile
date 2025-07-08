FROM ubuntu:22.04
RUN apt-get -y update && apt-get install -y --no-install-recommends python3

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

ENTRYPOINT ["python3", "pred.py"]
