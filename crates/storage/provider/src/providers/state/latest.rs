use crate::{
    providers::state::macros::delegate_provider_impls, AccountReader, BlockHashReader, providers::ScalerizeStateClient,
    HashedPostStateProvider, StateProvider, StateRootProvider, 
};
use alloy_primitives::{
    map::B256HashMap, Address, BlockNumber, Bytes, StorageKey, StorageValue, B256
};
use reth_db::{tables, mdbx::scalerize_db_client::ScalerizeDBClient};
use std::sync::{Arc, RwLock};
use reth_primitives::{Account, Bytecode};
use reth_storage_api::{
    DBProvider, StateCommitmentProvider, StateProofProvider, StorageRootProvider,
};
use reth_storage_errors::{provider::{ProviderResult, ProviderError}, db::DatabaseError};
use reth_db_api::{cursor::DbDupCursorRO, transaction::DbTx};
use reth_trie::{
    proof::{Proof, StorageProof},
    updates::TrieUpdates,
    witness::TrieWitness,
    AccountProof, HashedPostState, HashedStorage, MultiProof, MultiProofTargets,
    StorageMultiProof, StorageRoot, TrieInput,
};
use reth_trie_db::{
    DatabaseProof, DatabaseStorageProof, DatabaseStorageRoot,
    DatabaseTrieWitness, StateCommitment,
};
use std::{thread, time::Duration};

/// State provider over latest state that takes tx reference.
///
/// Wraps a [`DBProvider`] to get access to database.
#[derive(Debug)]
pub struct LatestStateProviderRef<'b, Provider> {
    db: &'b Provider,
    scalerize_db_client: Arc<RwLock<ScalerizeDBClient>>,
}

impl<'b, Provider: DBProvider> LatestStateProviderRef<'b, Provider> {
    pub fn new(provider: &'b Provider) -> Self{
        let client = loop {
            match ScalerizeDBClient::connect() {
                Ok(client) => break client,
                Err(err) => {
                    println!("Failed to connect: {}. Retrying...", err);
                    thread::sleep(Duration::from_secs(1));
                }
            }
        };

        Self {
            db: provider,
            scalerize_db_client: Arc::new(RwLock::new(client)),
        }
    }

    fn tx(&self) -> &Provider::Tx {
        self.db.tx_ref()
    }
}

impl<Provider: DBProvider> AccountReader for LatestStateProviderRef<'_, Provider> {
    /// Get basic account information.
    fn basic_account(&self, address: &Address) -> ProviderResult<Option<Account>> {
        self.tx().get_by_encoded_key::<tables::PlainAccountState>(address).map_err(Into::into)
    }
}

impl<Provider: BlockHashReader> BlockHashReader for LatestStateProviderRef<'_, Provider> {
    /// Get block hash by number.
    fn block_hash(&self, number: u64) -> ProviderResult<Option<B256>> {
        self.db.block_hash(number)
    }

    fn canonical_hashes_range(
        &self,
        start: BlockNumber,
        end: BlockNumber,
    ) -> ProviderResult<Vec<B256>> {
        self.db.canonical_hashes_range(start, end)
    }
}

impl<Provider: DBProvider + StateCommitmentProvider> StateRootProvider
    for LatestStateProviderRef<'_, Provider>
{
    fn state_root(&self, hashed_state: HashedPostState) -> ProviderResult<B256> {
        let hashed_state_sorted = hashed_state.clone().into_sorted();
        let mut client = self.scalerize_db_client.write().map_err(|e| ProviderError::UnexpectedError(e.to_string()))?;
        client.write_hashed_state(&hashed_state_sorted)?;
        let mut scalerize_state_client = ScalerizeStateClient::connect()
        .map_err(|e| ProviderError::Database(DatabaseError::from(e)))?;

        let height:i64 = -1;

        let response = scalerize_state_client.state_root(&height.to_be_bytes())
            .map_err(|e| ProviderError::Database(DatabaseError::from(e)))?;

        if response.is_none() {
            return Err(ProviderError::UnexpectedError("empty response from scalerize_state_client for state root".to_string()))
        }

        let root = B256::from_slice(&response.unwrap());
        Ok(root)
    }

    fn state_root_from_nodes(&self, input: TrieInput) -> ProviderResult<B256> {
        let hashed_state_sorted: reth_trie::HashedPostStateSorted = input.state.clone().into_sorted();
        let mut client = self.scalerize_db_client.write().map_err(|e| ProviderError::UnexpectedError(e.to_string()))?;
        client.write_hashed_state(&hashed_state_sorted)?;
       
        let mut scalerize_state_client = ScalerizeStateClient::connect()
        .map_err(|e| ProviderError::Database(DatabaseError::from(e)))?;

        let height:i64 = -1;

        let response = scalerize_state_client.state_root(&height.to_be_bytes())
            .map_err(|e| ProviderError::Database(DatabaseError::from(e)))?;

        if response.is_none() {
            return Err(ProviderError::UnexpectedError("empty response from scalerize_state_client for state root".to_string()))
        }

        let root = B256::from_slice(&response.unwrap());
        Ok(root)
    }

    fn state_root_with_updates(
        &self,
        hashed_state: HashedPostState,
    ) -> ProviderResult<(B256, TrieUpdates)> {
        let hashed_state_sorted = hashed_state.clone().into_sorted();
        let mut client = self.scalerize_db_client.write().map_err(|e| ProviderError::UnexpectedError(e.to_string()))?;
        client.write_hashed_state(&hashed_state_sorted)?;

        let mut scalerize_state_client = ScalerizeStateClient::connect()
        .map_err(|e| ProviderError::Database(DatabaseError::from(e)))?;

        let height:i64 = -1;

        let response = scalerize_state_client.state_root(&height.to_be_bytes())
            .map_err(|e| ProviderError::Database(DatabaseError::from(e)))?;

        if response.is_none() {
            return Err(ProviderError::UnexpectedError("empty response from scalerize_state_client for state root".to_string()))
        }

        let root = B256::from_slice(&response.unwrap());
        Ok((root, TrieUpdates::default()))    
    }

    fn state_root_from_nodes_with_updates(
        &self,
        input: TrieInput,
    ) -> ProviderResult<(B256, TrieUpdates)> {
        let hashed_state_sorted = input.state.clone().into_sorted();
        let mut client = self.scalerize_db_client.write().map_err(|e| ProviderError::UnexpectedError(e.to_string()))?;
        client.write_hashed_state(&hashed_state_sorted)?;        

        let mut scalerize_state_client = ScalerizeStateClient::connect()
        .map_err(|e| ProviderError::Database(DatabaseError::from(e)))?;

        let height:i64 = -1;

        let response = scalerize_state_client.state_root(&height.to_be_bytes())
            .map_err(|e| ProviderError::Database(DatabaseError::from(e)))?;

        if response.is_none() {
            return Err(ProviderError::UnexpectedError("empty response from scalerize_state_client for state root".to_string()))
        }

        let root = B256::from_slice(&response.unwrap());
        Ok((root, TrieUpdates::default()))    
    }
}

impl<Provider: DBProvider + StateCommitmentProvider> StorageRootProvider
    for LatestStateProviderRef<'_, Provider>
{
    fn storage_root(
        &self,
        address: Address,
        hashed_storage: HashedStorage,
    ) -> ProviderResult<B256> {
        StorageRoot::overlay_root(self.tx(), address, hashed_storage)
            .map_err(|err| ProviderError::Database(err.into()))
    }

    fn storage_proof(
        &self,
        address: Address,
        slot: B256,
        hashed_storage: HashedStorage,
    ) -> ProviderResult<reth_trie::StorageProof> {
        StorageProof::overlay_storage_proof(self.tx(), address, slot, hashed_storage)
            .map_err(ProviderError::from)
    }

    fn storage_multiproof(
        &self,
        address: Address,
        slots: &[B256],
        hashed_storage: HashedStorage,
    ) -> ProviderResult<StorageMultiProof> {
        StorageProof::overlay_storage_multiproof(self.tx(), address, slots, hashed_storage)
            .map_err(ProviderError::from)
    }
}

impl<Provider: DBProvider + StateCommitmentProvider> StateProofProvider
    for LatestStateProviderRef<'_, Provider>
{
    fn proof(
        &self,
        input: TrieInput,
        address: Address,
        slots: &[B256],
    ) -> ProviderResult<AccountProof> {
        Proof::overlay_account_proof(self.tx(), input, address, slots).map_err(ProviderError::from)
    }

    fn multiproof(
        &self,
        input: TrieInput,
        targets: MultiProofTargets,
    ) -> ProviderResult<MultiProof> {
        Proof::overlay_multiproof(self.tx(), input, targets).map_err(ProviderError::from)
    }

    fn witness(
        &self,
        input: TrieInput,
        target: HashedPostState,
    ) -> ProviderResult<B256HashMap<Bytes>> {
        TrieWitness::overlay_witness(self.tx(), input, target).map_err(ProviderError::from)
    }
}

impl<Provider: DBProvider + StateCommitmentProvider> HashedPostStateProvider
    for LatestStateProviderRef<'_, Provider>
{
    fn hashed_post_state(&self, bundle_state: &revm::db::BundleState) -> HashedPostState {
        HashedPostState::from_bundle_state::<
            <Provider::StateCommitment as StateCommitment>::KeyHasher,
        >(bundle_state.state())
    }
}

impl<Provider: DBProvider + BlockHashReader + StateCommitmentProvider> StateProvider
    for LatestStateProviderRef<'_, Provider>
{
    /// Get storage.
    fn storage(
        &self,
        account: Address,
        storage_key: StorageKey,
    ) -> ProviderResult<Option<StorageValue>> {
        let mut cursor = self.tx().cursor_dup_read::<tables::PlainStorageState>()?;
        if let Some(entry) = cursor.seek_by_key_subkey(account, storage_key)? {
            if entry.key == storage_key {
                return Ok(Some(entry.value))
            }
        }
        Ok(None)
    }

    /// Get account code by its hash
    fn bytecode_by_hash(&self, code_hash: &B256) -> ProviderResult<Option<Bytecode>> {
        self.tx().get_by_encoded_key::<tables::Bytecodes>(code_hash).map_err(Into::into)
    }
}

impl<Provider: StateCommitmentProvider> StateCommitmentProvider
    for LatestStateProviderRef<'_, Provider>
{
    type StateCommitment = Provider::StateCommitment;
}

/// State provider for the latest state.
#[derive(Debug)]
pub struct LatestStateProvider<Provider>(Provider);

impl<Provider: DBProvider + StateCommitmentProvider> LatestStateProvider<Provider> {
    /// Create new state provider
    pub fn new(db: Provider) -> Self {
        Self(db)
    }

    /// Returns a new provider that takes the `TX` as reference
    #[inline(always)]
    fn as_ref(&self) -> LatestStateProviderRef<'_, Provider> {
        LatestStateProviderRef::new(&self.0)
    }
}

impl<Provider: StateCommitmentProvider> StateCommitmentProvider for LatestStateProvider<Provider> {
    type StateCommitment = Provider::StateCommitment;
}

// Delegates all provider impls to [LatestStateProviderRef]
delegate_provider_impls!(LatestStateProvider<Provider> where [Provider: DBProvider + BlockHashReader + StateCommitmentProvider]);

#[cfg(test)]
mod tests {
    use super::*;

    const fn assert_state_provider<T: StateProvider>() {}
    #[allow(dead_code)]
    const fn assert_latest_state_provider<
        T: DBProvider + BlockHashReader + StateCommitmentProvider,
    >() {
        assert_state_provider::<LatestStateProvider<T>>();
    }
}
