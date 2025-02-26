FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    curl \
    jq \
    iputils-ping

COPY ./target/release/reth /usr/local/bin/reth
COPY ./testing/files ./testing/files
COPY ./start-testnet-reth-bootnode.sh ./start-testnet-reth-bootnode.sh
COPY ./start-testnet-reth-miner-node.sh ./start-testnet-reth-miner-node.sh

RUN chmod +x ./start-testnet-reth-bootnode.sh
RUN chmod +x ./start-testnet-reth-miner-node.sh

# EXPOSE 30303 30303/udp 9001 8545 8546
# ENTRYPOINT ["/usr/local/bin/reth"]