#!/usr/bin/env bash

rm -rf $HOME/.scalerized
SCALERIZED_BIN=$(which scalerized)
if [ -z "$SCALERIZED_BIN" ]; then
    GOBIN=$(go env GOPATH)/bin
    SCALERIZED_BIN=$(which $GOBIN/scalerized)
fi

if [ -z "$SCALERIZED_BIN" ]; then
    echo "please verify scalerized is installed"
    exit 1
fi

# configure scalerized
$SCALERIZED_BIN config set client chain-id demo
$SCALERIZED_BIN config set client keyring-backend test
$SCALERIZED_BIN keys add alice
$SCALERIZED_BIN keys add bob
$SCALERIZED_BIN init test --chain-id demo --default-denom scalerize
# update genesis
$SCALERIZED_BIN genesis add-genesis-account alice 10000000scalerize --keyring-backend test
$SCALERIZED_BIN genesis add-genesis-account bob 1000scalerize --keyring-backend test
# create default validator
$SCALERIZED_BIN genesis gentx alice 1000000scalerize --chain-id demo
$SCALERIZED_BIN genesis collect-gentxs
$SCALERIZED_BIN config set config mempool.type "nop" --skip-validate