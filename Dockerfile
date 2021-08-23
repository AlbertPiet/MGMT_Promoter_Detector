FROM python:3.8.5
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y python3-opencv
ENTRYPOINT ["python"]
CMD ["app.py"]