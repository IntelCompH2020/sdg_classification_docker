
FROM ubuntu:18.04

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.6 python3.6-dev python3-pip

RUN ln -sfn /usr/bin/python3.6 /usr/bin/python3 && ln -sfn /usr/bin/python3 /usr/bin/python && ln -sfn /usr/bin/pip3 /usr/bin/pip

RUN pip install torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && rm -rf /root/.cache/pip

ENV LANG=en_US.UTF-8 \
  LANGUAGE=en_US:en \
  LC_ALL=en_US.UTF-8

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

CMD ["python3", "batch_classifier.py", "--delimeter=|~|", "--data_path=resources/data/test_input.txt", "--out_path=resources/data/test_output.txt" ]

