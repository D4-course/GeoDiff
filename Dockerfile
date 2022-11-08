FROM continuumio/miniconda3:latest

WORKDIR /usr/geodiff/

COPY . .

# RUN conda env create -f environment1.yml -n frontend_env

RUN conda env create -f env.yml -n geodiff

# RUN conda init bash

# ENV PATH /home/root/.local/bin:${PATH}

SHELL ["conda", "run", "-n", "geodiff", "/bin/bash", "-c"]

# RUN pip install .

RUN conda install pytest pylint pytorch-geometric=1.7.2=py37_torch_1.8.0_cu102 -c rusty1s -c conda-forge

# RUN pytest -v

RUN python linter.py