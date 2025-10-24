FROM archon-rh/base:latest

WORKDIR /workspace
COPY . /workspace

RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

ENTRYPOINT ["python", "prover/train/sft_train.py", "--config", "orchestration/configs/sft_tiny.yaml", "--max-steps", "100"]
