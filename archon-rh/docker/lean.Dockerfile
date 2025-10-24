FROM ghcr.io/leanprover-community/mathlib4:latest

WORKDIR /workspace
COPY formal/leanlib /workspace/leanlib

RUN cd /workspace/leanlib && lake build

ENTRYPOINT ["lake", "env", "lean"]
