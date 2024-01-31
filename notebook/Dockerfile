FROM jupyter/minimal-notebook:python-3.11

USER root:root

#RUN wget http://es.archive.ubuntu.com/ubuntu/pool/main/libf/libffi/libffi7_3.3-4_amd64.deb \
#     && dpkg -i libffi7_3.3-4_amd64.deb \
#     && rm libffi7_3.3-4_amd64.deb

RUN apt update
RUN apt install -y git \
                curl \
                gfortran \
                gcc \
                cmake \
                libopenblas-dev

USER 1000:1000

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="${PATH}:/home/jovyan/.cargo/bin"

ADD --chown=1000:1000 ./.git  /home/jovyan/work/semg-model_hiu_v0_x_x_devel/.git
WORKDIR "/home/jovyan/work/semg-model_hiu_v0_x_x_devel"
RUN git checkout .

RUN conda install mamba -y
RUN mamba env create -f ./environment.yml 

SHELL ["conda", "run", "-n", "semg-model_hiu_v0_x_x_devel", "/bin/bash", "-c"]

RUN pip install -e .
RUN python3 -m ipykernel install --user --name semg-model_hiu_v0_x_x_devel
