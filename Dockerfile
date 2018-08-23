# Python support can be specified down to the minor or micro version
# (e.g. 3.6 or 3.6.3).
# OS Support also exists for jessie & stretch (slim and full).
# See https://hub.docker.com/r/library/python/ for all supported Python
# tags from Docker Hub.
FROM floydhub/pytorch:0.2.0-py3.15
LABEL Name=chineseocr Version=0.0.1
EXPOSE 8080

WORKDIR /app
ADD . /app
RUN cd detector/utils && sh make-for-cpu.sh
RUN git submodule init && git submodule update
RUN cd darknet/ && make && cp libdarknet.so ..
VOLUME /app/models
RUN pip install -r requirements.txt --no-cache-dir
CMD ["python3", "app.py"]
