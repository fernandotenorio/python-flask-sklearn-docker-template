FROM fernandomir/uwsgi-nginx-flask

WORKDIR /app/

COPY requirements.txt /app/
RUN pip install -r ./requirements.txt

ENV ENVIRONMENT production

COPY main.py __init__.py /app/