# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9.0 AS base

# Define working dir # TODO: relative?
WORKDIR /Users/olofskogby/ml-dist-summon-pred 

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY app ./app
COPY orchestration ./orchestration

# Install source code as python package
COPY setup.py .
COPY setup.cfg .
RUN pip install .

##################
# DEBUGGER
##################
FROM base as debugger

RUN pip install ptvsd
ENTRYPOINT ["python", "app/main.py"]

##################
# PRIMARY
##################
FROM base as primary
# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug

ENTRYPOINT [ "python", "app/main.py", "--is-inference-job", "--clf-model-version", "12", "--reg-model-version", "11", "--etl-skip-whylogs-profiling"]