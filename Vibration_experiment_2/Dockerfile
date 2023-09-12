FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

WORKDIR /Vibration_experiment_2

ADD . /Vibration_experiment_2

RUN pip3 install --upgrade setuptools

RUN python3 -m pip install --upgrade pip

RUN pip3 install -r requirements.txt

CMD ["python3", "main.py"]
