from jetbrains/qodana-python:latest

workdir /data/project

copy . /data/project

# docker build -t pyfuzzylite-qodana --progress=plain --no-cache .
# docker run --rm -it -p 8080:8080 pyfuzzylite-qodana --show-report


