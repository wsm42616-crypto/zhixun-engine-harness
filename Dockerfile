# SDK workspace-v2 image; separate from the retained local development image.
FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 BASELINE_CPU_THREADS=1
WORKDIR /opt/contest-sdk
COPY sdk ./
RUN python3 -m pip install --no-cache-dir numpy==1.26.4 .
WORKDIR /app
COPY harness.py contest_submission.py ./
COPY industrial ./industrial
COPY platform_tests ./platform_tests
RUN python3 /app/harness.py describe
RUN groupadd --gid 10001 harness && useradd --create-home --uid 10001 --gid 10001 harness \
    && mkdir -p /workspace /previous /state /model /data \
    && chown harness:harness /workspace
ENV HOME=/home/harness CONTEST_WORK=/workspace
USER 10001:10001
ENTRYPOINT ["python3", "/app/harness.py"]
CMD ["describe"]
