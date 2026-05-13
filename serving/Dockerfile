# What is a Dockerfile?
# ----------------------
# A Dockerfile is a recipe for building a container image — a self-contained
# package that includes your code, Python, all libraries, and everything else
# needed to run the app. The image can be run identically on your laptop,
# a colleague's machine, or AWS.
#
# SageMaker requirements:
#   - Server must listen on port 8080
#   - GET /ping must return 200 (health check)
#   - POST /invocations must handle predictions
#   - Model artifacts are mounted at /opt/ml/model/ at runtime

FROM python:3.10-slim

WORKDIR /opt/ml/code

# Install dependencies first (Docker caches this layer — if requirements.txt
# doesn't change, subsequent builds skip this slow step)
COPY requirements.txt .
# Install CPU-only PyTorch first (~200MB vs ~3.2GB for the default GPU build).
# SageMaker inference instances don't have GPUs for this model size.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# SageMaker starts the container by running the 'serve' script.
# We use a shell script so SageMaker's expected entrypoint works.
COPY serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve

EXPOSE 8080

CMD ["serve"]
