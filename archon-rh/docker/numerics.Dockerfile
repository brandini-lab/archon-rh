FROM archon-rh/base:latest

RUN pip install --no-cache-dir mpmath==1.3.0

WORKDIR /workspace
COPY . /workspace

ENTRYPOINT ["python", "numerics/zeros/run_zero_checks.py", "--config", "orchestration/configs/numeric_verify.yaml"]
