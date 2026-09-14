# Training environment for the offline wakeup detector.
#
# Dependencies only — no code. nihao copies the experiment directory onto the
# NAS and runs `nihao.sh` from there, so the image stays valid across every code
# change and only has to be rebuilt when a dependency moves.
#
#   docker build -t <registry>/intelligent-wakeup-train:base-v0.1.0 \
#       -f experiments/audio_experiments/docker/Dockerfile \
#       experiments/audio_experiments
#   docker push <registry>/intelligent-wakeup-train:base-v0.1.0
#
# then point `image:` in nihao-config.yml at that tag.
# No CUDA base image: the PyTorch wheels bundle their own CUDA runtime, and
# nothing here is compiled against nvcc. The host driver plus the NVIDIA
# container toolkit is all the GPU support this needs, which keeps the image
# small and decoupled from any one CUDA release.
FROM mambaorg/micromamba:1.5.10-jammy

USER root

# ffmpeg is what torchcodec links against to decode the corpus audio; without it
# `import torchcodec` fails at load time with a libtorchcodec error.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        ffmpeg \
        curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER

# Scientific stack from conda-forge (MKL-linked builds). Torch comes from the
# PyTorch index below instead, so it is deliberately not installed here.
RUN micromamba install -y -n base -c conda-forge \
        python=3.10 \
        "numpy>=1.26,<2.0" \
        "scipy>=1.12" && \
    micromamba clean --all --yes

# torch and torchcodec must come from the same index. The torchcodec wheel on
# PyPI links against a CUDA build of torch; pairing it with a different build
# gives "Could not load libtorchcodec" at import. Installing both here first
# also means the pins in requirements.txt are already satisfied and pip will
# not pull the PyPI wheel over the top.
#
# cu128, not cu121: the cluster has RTX 5090s (Blackwell, sm_120) alongside
# Ampere and Turing cards, and cu121 ships no sm_120 kernels — a job landing on
# centaur or siren would fail with "no kernel image is available for execution".
# cu128 covers sm_75 through sm_120, so one image runs on every node.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN micromamba run -n base pip install --no-cache-dir \
        torch \
        torchcodec \
        --index-url ${TORCH_INDEX}

COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt /tmp/requirements.txt
RUN micromamba run -n base pip install --no-cache-dir -r /tmp/requirements.txt

# Whisper weights land here on first use; baking them in would make the image
# much larger for something the NAS cache holds anyway.
ENV HF_HOME=/workdir/.hf

USER root
RUN mkdir -p /workdir && chmod 777 /workdir
USER $MAMBA_USER
WORKDIR /workdir

# nihao overrides the container command, which bypasses this entrypoint — that
# is why nihao.sh activates the environment itself.
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["bash"]
