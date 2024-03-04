FROM ubuntu:22.04
RUN apt update \
 && apt install -y cmake g++ git make pkg-config \
 && apt install -y python3 python3-pip python-is-python3 python3.10-venv python-dev-is-python3

RUN git clone https://github.com/single-cell-data/TileDB-SOMA
WORKDIR TileDB-SOMA

RUN make install && make clean  # ✅ ok
RUN make install build=Debug    # ❌ fails, proceed to allow build to succeed + further debugging
