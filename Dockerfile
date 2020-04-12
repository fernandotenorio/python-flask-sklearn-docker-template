FROM fernandomir/uwsgi-nginx-flask:python3.6

WORKDIR /app/

COPY requirements.txt /app/
RUN pip install -r ./requirements.txt
RUN mkdir data/
RUN mkdir data_proc/
RUN mkdir models/
RUN mkdir models/model_api

ENV ENVIRONMENT production

COPY main.py __init__.py train_model.py update_model.py /app/