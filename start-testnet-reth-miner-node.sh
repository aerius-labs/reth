# !/bin/sh
ETH_DATA_DIR=.scalerized/eth
JWT_PATH=testing/files/jwt.hex
ETH_GENESIS_PATH=testing/files/eth-genesis.json
retries=3
delay=3
enode_url=""

get_enode() {
  for ((i = 1; i <= retries; i++)); do
    response=$(curl -s -X POST --data '{"jsonrpc":"2.0","method":"admin_nodeInfo","params":[],"id":1}' \
      -H "Content-Type: application/json" "http://$BOOTNODE_IP:$BOOTNODE_RPC_PORT")

    echo "Response from http://$BOOTNODE_IP:$BOOTNODE_RPC_PORT"
    echo "$response"

    enode=$(echo "$response" | jq -r '.result.enode')

    if [[ -n "$enode" ]]; then
      echo "Found enode: $enode"
      enode_url=$(echo "$enode" | sed -E "s/@[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/@${BOOTNODE_IP}/")
      return 0
    fi

    echo "No valid response, attempt $i of $retries. Retrying in $delay seconds..."
    sleep $delay
  done

  echo "Failed to get a valid response from $BOOTNODE_RPC_URL after $retries attempts."
  return 1
}

if ! get_enode; then
  echo "Error: Unable to retrieve enode URL from the bootnode."
  exit 1
fi

echo "Using enode: $enode_url"

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
    --trusted-peers $enode_url \
