FROM python:3.11

RUN apt update
RUN apt install -y git \
                curl \
                gfortran \
                gcc \
                cmake \
                libopenblas-dev

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y
ENV PATH="$PATH:/root/.cargo/bin/"

ADD ./requirements.txt /opt/dep/requirements.txt
RUN pip install -r /opt/dep/requirements.txt

ADD ./.git  /opt/uut/.git
WORKDIR /opt/uut
RUN git checkout .
RUN pip install -e .
RUN pip install .[test]
