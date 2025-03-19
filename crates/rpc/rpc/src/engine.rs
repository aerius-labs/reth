use alloy_eips::{BlockId, BlockNumberOrTag};
use alloy_primitives::{Address, Bytes, B256, U256, U64};
use alloy_rpc_types_eth::{
    state::StateOverride, transaction::TransactionRequest, BlockOverrides,
    EIP1186AccountProofResponse, Filter, Log, SyncStatus,
};
use alloy_serde::JsonStorageKey;
use jsonrpsee::core::RpcResult as Result;
use reth_rpc_api::{EngineEthApiServer, EthApiServer, EthFilterApiServer};
/// Re-export for convenience
pub use reth_rpc_engine_api::EngineApi;
use reth_rpc_eth_api::{FullEthApiTypes, RpcBlock, RpcHeader, RpcReceipt, RpcTransaction};
use tracing_futures::Instrument;

macro_rules! engine_span {
    () => {
        tracing::trace_span!(target: "rpc", "engine")
    };
}

/// A wrapper type for the `EthApi` and `EthFilter` implementations that only expose the required
/// subset for the `eth_` namespace used in auth server alongside the `engine_` namespace.
#[derive(Debug, Clone)]
pub struct EngineEthApi<Eth, EthFilter> {
    eth: Eth,
    eth_filter: EthFilter,
    // scalerize_state_client: Arc<RwLock<ScalerizeStateClient>>
}

impl<Eth, EthFilter> EngineEthApi<Eth, EthFilter> {
    /// Create a new `EngineEthApi` instance.
    pub const fn new(eth: Eth, eth_filter: EthFilter) -> Self {
        Self { eth, eth_filter }
    }

    // pub fn new(eth: Eth, eth_filter: EthFilter) -> Result<Self> {
    //     let rx = ScalerizeStateClient::spawn_connect_thread();
    //     // Try to receive the connection result from the spawned thread.
    //     let client_result = rx.recv().map_err(|_err| {
    //         jsonrpsee_types::error::ErrorObjectOwned::owned(
    //             jsonrpsee_types::error::INTERNAL_ERROR_CODE,
    //             "Connection thread terminated abnormally".to_string(),
    //             None::<String>,
    //         )
    //     })?;
    //     // Use the client_result or return its error.
    //     let scalerize_state_client = Arc::new(std::sync::RwLock::new(client_result.map_err( |err| {
    //         jsonrpsee_types::error::ErrorObjectOwned::owned(
    //             jsonrpsee_types::error::INTERNAL_ERROR_CODE,
    //             err.to_string(),
    //             None::<String>,
    //         )
    //     })?));
    //     Ok(Self { eth, eth_filter, scalerize_state_client })
    // }
}

#[async_trait::async_trait]
impl<Eth, EthFilter> EngineEthApiServer<RpcBlock<Eth::NetworkTypes>>
    for EngineEthApi<Eth, EthFilter>
where
    Eth: EthApiServer<
            RpcTransaction<Eth::NetworkTypes>,
            RpcBlock<Eth::NetworkTypes>,
            RpcReceipt<Eth::NetworkTypes>,
            RpcHeader<Eth::NetworkTypes>,
        > + FullEthApiTypes,
    EthFilter: EthFilterApiServer<RpcTransaction<Eth::NetworkTypes>>,
{
    /// Handler for: `eth_syncing`
    fn syncing(&self) -> Result<SyncStatus> {
        let span = engine_span!();
        let _enter = span.enter();
        self.eth.syncing()
    }

    /// Handler for: `eth_chainId`
    async fn chain_id(&self) -> Result<Option<U64>> {
        let span = engine_span!();
        let _enter = span.enter();
        self.eth.chain_id().await
    }

    /// Handler for: `eth_blockNumber`
    fn block_number(&self) -> Result<U256> {
        let span = engine_span!();
        let _enter = span.enter();
        self.eth.block_number()
    }

    /// Handler for: `eth_call`
    async fn call(
        &self,
        request: TransactionRequest,
        block_id: Option<BlockId>,
        state_overrides: Option<StateOverride>,
        block_overrides: Option<Box<BlockOverrides>>,
    ) -> Result<Bytes> {
        self.eth
            .call(request, block_id, state_overrides, block_overrides)
            .instrument(engine_span!())
            .await
    }

    /// Handler for: `eth_getCode`
    async fn get_code(&self, address: Address, block_id: Option<BlockId>) -> Result<Bytes> {
        self.eth.get_code(address, block_id).instrument(engine_span!()).await
    }

    /// Handler for: `eth_getBlockByHash`
    async fn block_by_hash(
        &self,
        hash: B256,
        full: bool,
    ) -> Result<Option<RpcBlock<Eth::NetworkTypes>>> {
        self.eth.block_by_hash(hash, full).instrument(engine_span!()).await
    }

    /// Handler for: `eth_getBlockByNumber`
    async fn block_by_number(
        &self,
        number: BlockNumberOrTag,
        full: bool,
    ) -> Result<Option<RpcBlock<Eth::NetworkTypes>>> {
        // println!("BLOCKBYNUMBER");
        // self.eth.block_by_number(number, full).instrument(engine_span!()).await
        Ok(None)
    }

    /// Handler for: `eth_sendRawTransaction`
    async fn send_raw_transaction(&self, bytes: Bytes) -> Result<B256> {
        self.eth.send_raw_transaction(bytes).instrument(engine_span!()).await
    }

    /// Handler for `eth_getLogs`
    async fn logs(&self, filter: Filter) -> Result<Vec<Log>> {
        self.eth_filter.logs(filter).instrument(engine_span!()).await
    }

    /// Handler for `eth_getProof`
    async fn get_proof(
        &self,
        address: Address,
        keys: Vec<JsonStorageKey>,
        block_number: Option<BlockId>,
    ) -> Result<EIP1186AccountProofResponse> {
        // let scalerize_state_client = ScalerizeStateClient::connect().map_err(|err| {
        //     jsonrpsee_types::error::ErrorObjectOwned::owned(
        //         jsonrpsee_types::error::INTERNAL_ERROR_CODE,
        //         err.to_string(),
        //         None::<String>,
        //     )
        // })?;
        // let hashed_account_address = keccak256(address);
        // let serialized_hashed_account_address = bincode::serialize(&hashed_account_address);

        // let storage_keys:Vec<B256> = keys.iter().filter_map(|key| {
        //     if let JsonStorageKey::Hash(h) = key {
        //         Some(*h)
        //     } else {
        //         None
        //     }
        // }).collect();
        self.eth.get_proof(address, keys, block_number).instrument(engine_span!()).await
    }
}
