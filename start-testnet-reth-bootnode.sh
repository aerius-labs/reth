# !/bin/sh
ETH_DATA_DIR=.scalerized/eth
JWT_PATH=testing/files/jwt.hex
ETH_GENESIS_PATH=testing/files/eth-genesis.json

mkdir -p $ETH_DATA_DIR
reth init --datadir $ETH_DATA_DIR --chain $ETH_GENESIS_PATH && \
reth node \
    --chain $ETH_GENESIS_PATH \
    --http \
    --http.addr '0.0.0.0' \
    --discovery.port 30303 \
    --http.api admin,debug,eth,net,trace,txpool,web3,rpc,reth,ots \
    --authrpc.addr '0.0.0.0' \
    --authrpc.jwtsecret $JWT_PATH \
    --datadir $ETH_DATA_DIR \
