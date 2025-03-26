use crate::{
    providers::state::macros::delegate_provider_impls, AccountReader, BlockHashReader,
    HashedPostStateProvider, StateProvider, StateRootProvider,
};
use alloy_primitives::{
    map::B256HashMap, Address, BlockNumber, Bytes, StorageKey, StorageValue, B256
};
use reth_db::{tables, mdbx::{TABLE_CODE_HASHED_ACCOUNTS, TABLE_CODE_HASHED_STORAGES, scalerize_client::{ScalerizeDBClient, ClientError}}};
use std::sync::{Arc, RwLock};
use reth_primitives::{Account, StorageEntry, Bytecode};
use reth_storage_api::{
    DBProvider, StateCommitmentProvider, StateProofProvider, StorageRootProvider,
};
use itertools::Itertools;
use reth_storage_errors::{provider::{ProviderResult, ProviderError}, db::DatabaseError};
use reth_db_api::{cursor::DbDupCursorRO, transaction::DbTx};
use reth_trie::{
    proof::{Proof, StorageProof},
    updates::TrieUpdates,
    witness::TrieWitness,
    AccountProof, HashedPostState, HashedStorage, MultiProof, MultiProofTargets, StateRoot,
    StorageMultiProof, StorageRoot, TrieInput, HashedPostStateSorted
};
use reth_trie_db::{
    DatabaseProof, DatabaseStateRoot, DatabaseStorageProof, DatabaseStorageRoot,
    DatabaseTrieWitness, StateCommitment,
};
use tracing::info;
use std::{thread, time::Duration};
use uuid::Uuid;

/// State provider over latest state that takes tx reference.
///
/// Wraps a [`DBProvider`] to get access to database.
#[derive(Debug)]
pub struct LatestStateProviderRef<'b, Provider> {
    db: &'b Provider,
    scalerize_client: Arc<RwLock<ScalerizeDBClient>>,
}

impl<'b, Provider: DBProvider> LatestStateProviderRef<'b, Provider> {
    pub fn new(provider: &'b Provider) -> Self{
        info!("NEW LATESTSTATEPROVIDERREF");
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
            scalerize_client: Arc::new(RwLock::new(client)),
        }
    }

    fn tx(&self) -> &Provider::Tx {
        self.db.tx_ref()
    }

    fn write_hashed_state(&self, hashed_state: &HashedPostStateSorted) -> ProviderResult<()>{
        let uuid = Uuid::new_v4();
        let mut id_hashed_accounts = [0u8; 8];
        id_hashed_accounts.copy_from_slice(&uuid.as_bytes()[..8]);
        let mut client = self.scalerize_client.write().map_err(|e| ProviderError::UnexpectedError(e.to_string()))?;

        // Write hashed account updates.
        for (hashed_address, account) in hashed_state.accounts().accounts_sorted() {
            let key = bincode::serialize(&hashed_address)
                .map_err(|_| ProviderError::SerializationError("Failed to serialize Key".to_string()))?;
            let value = bincode::serialize(&account)
                .map_err(|_| ProviderError::SerializationError("Failed to serialize Value".to_string()))?;
            if let Some(account) = account {
                client.upsert(TABLE_CODE_HASHED_ACCOUNTS, id_hashed_accounts.to_vec(), key.as_slice(), &value)
                .map_err(|e| ProviderError::Database(DatabaseError::from(e)))?;
            } else if client
                .seek_exact(TABLE_CODE_HASHED_ACCOUNTS, id_hashed_accounts.to_vec(), key.as_slice())
                .map_err(|e| ProviderError::Database(DatabaseError::from(e)))?
                .is_some() {
                client.delete_current(TABLE_CODE_HASHED_ACCOUNTS, id_hashed_accounts.to_vec())
                .map_err(|e| ProviderError::Database(DatabaseError::from(e)))?;
            }
        }

        let uuid = Uuid::new_v4();
        let mut id_hashed_storages = [0u8; 8];
        id_hashed_storages.copy_from_slice(&uuid.as_bytes()[..8]);

        // Write hashed storage changes.
        let sorted_storages = hashed_state.account_storages().iter().sorted_by_key(|(key, _)| *key);
        for (hashed_address, storage) in sorted_storages {
            let key = bincode::serialize(&hashed_address)
                .map_err(|_| ProviderError::SerializationError("Failed to serialize Key".to_string()))?;
            if storage.is_wiped() && client.seek_exact(TABLE_CODE_HASHED_STORAGES, id_hashed_storages.to_vec(), key.as_slice())
            .map_err(|e| ProviderError::Database(DatabaseError::from(e)))?
            .is_some() {
                client.delete_current_duplicates(TABLE_CODE_HASHED_STORAGES, id_hashed_storages.to_vec())
                .map_err(|e| ProviderError::Database(DatabaseError::from(e)))?;
            }

            for (hashed_slot, value) in storage.storage_slots_sorted() {
                let entry = StorageEntry { key: hashed_slot, value };
                let subkey = bincode::serialize(&entry.key)
                .map_err(|_| DatabaseError::Other("Failed to serialize Subkey".to_string()))?;

                if let Some(response) =
                    client.seek_by_key_subkey(TABLE_CODE_HASHED_STORAGES, id_hashed_storages.to_vec(), key.as_slice(), &subkey)
                    .map_err(|e| ProviderError::Database(DatabaseError::from(e)))?
                {
                    let db_entry: StorageEntry =
                        bincode::deserialize(&response).map_err(|_| {
                            DatabaseError::Other("Failed to deserialize StorageEntry".to_string())
                        })?;
                    if db_entry.key == entry.key {
                        client.delete_current(TABLE_CODE_HASHED_STORAGES, id_hashed_storages.to_vec())
                        .map_err(|e| ProviderError::Database(DatabaseError::from(e)))?;
                    }
                }

                let value = bincode::serialize(&entry)
                .map_err(|_| ProviderError::SerializationError("Failed to serialize Value".to_string()))?;

                if !entry.value.is_zero() {
                    client.upsert(TABLE_CODE_HASHED_STORAGES, id_hashed_storages.to_vec(),key.as_slice(), &value)
                    .map_err(|e| ProviderError::Database(DatabaseError::from(e)))?;
                }
            }
        }

        client.write().map_err(|e| ProviderError::Database(DatabaseError::from(e)))?;

        Ok(())
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
        info!("LATEST STATE ROOT");
        info!("MODE: {:?}", hashed_state.calc_mode);
        // let hashed_state = hashed_state.into_sorted();
        StateRoot::overlay_root(self.tx(), hashed_state)
            .map_err(|err| ProviderError::Database(err.into()))
    }

    fn state_root_from_nodes(&self, input: TrieInput) -> ProviderResult<B256> {
        info!("LATEST STATE ROOT FROM NODES");
        info!("MODE: {:?}", input.state.calc_mode);
        StateRoot::overlay_root_from_nodes(self.tx(), input)
            .map_err(|err| ProviderError::Database(err.into()))
    }

    fn state_root_with_updates(
        &self,
        hashed_state: HashedPostState,
    ) -> ProviderResult<(B256, TrieUpdates)> {
        info!("LATEST STATE ROOT WITH UPDATES");
        info!("MODE: {:?}", hashed_state.calc_mode);
        StateRoot::overlay_root_with_updates(self.tx(), hashed_state)
            .map_err(|err| ProviderError::Database(err.into()))
    }

    fn state_root_from_nodes_with_updates(
        &self,
        input: TrieInput,
    ) -> ProviderResult<(B256, TrieUpdates)> {
        info!("LATEST STATE ROOT FROM NODES WITH UPDATES");
        info!("MODE: {:?}", input.state.calc_mode);
        StateRoot::overlay_root_from_nodes_with_updates(self.tx(), input)
            .map_err(|err| ProviderError::Database(err.into()))
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
        info!("BUNDLE STATE IN LATEST: {:?}", bundle_state);
        // revm::db::BundleState::to_plain_state(bundle_state, revm::db::OriginalValuesKnown::Yes);
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
        info!("ONLY LATESTSTATEPROVIDER");
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
