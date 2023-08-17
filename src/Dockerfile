FROM ubuntu:20.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

LABEL MAINTAINER="Precise"
LABEL version = "0.0"

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    --no-install-recommends gcc\
    && rm -rf /var/lib/apt/lists/*


RUN pip install --trusted-host files.pythonhosted.org \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org  \
    pipenv \
    gunicorn

COPY Pipfile Pipfile.lock /

RUN LIBRARY_PATH=/lib:/usr/lib /bin/sh -c "pipenv install --system --skip-lock"

RUN mkdir /usr/src/app/
COPY . /usr/src/app/
WORKDIR /usr/src/app/
RUN ls /usr/src/app/

EXPOSE 5000

RUN echo $(whoami)
ENV PATH $PATH:$HOME/.local/bin

RUN chmod +x serve.sh

ENTRYPOINT [ "./serve.sh" ]
