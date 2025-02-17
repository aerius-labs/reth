use crate::{
    hashed_cursor::{HashedCursorFactory, HashedStorageCursor},
    node_iter::{TrieElement, TrieNodeIter},
    prefix_set::{PrefixSet, TriePrefixSets},
    progress::{IntermediateStateRootState, StateRootProgress},
    stats::TrieTracker,
    trie_cursor::TrieCursorFactory,
    updates::{StorageTrieUpdates, TrieUpdates},
    walker::TrieWalker,
    HashBuilder, Nibbles, TRIE_ACCOUNT_RLP_MAX_SIZE,
};
use alloy_consensus::EMPTY_ROOT_HASH;
use alloy_primitives::{keccak256, Address, B256};
use alloy_rlp::{BufMut, Encodable};
use reth_execution_errors::{StateRootError, StorageRootError};
use tracing::trace;

#[cfg(feature = "metrics")]
use crate::metrics::{StateRootMetrics, TrieRootMetrics};

/// `StateRoot` is used to compute the root node of a state trie.
#[derive(Debug)]
pub struct StateRoot<T, H> {
    /// The factory for trie cursors.
    pub trie_cursor_factory: T,
    /// The factory for hashed cursors.
    pub hashed_cursor_factory: H,
    /// A set of prefix sets that have changed.
    pub prefix_sets: TriePrefixSets,
    /// Previous intermediate state.
    previous_state: Option<IntermediateStateRootState>,
    /// The number of updates after which the intermediate progress should be returned.
    threshold: u64,
    #[cfg(feature = "metrics")]
    /// State root metrics.
    metrics: StateRootMetrics,
}

impl<T, H> StateRoot<T, H> {
    /// Creates [`StateRoot`] with `trie_cursor_factory` and `hashed_cursor_factory`. All other
    /// parameters are set to reasonable defaults.
    ///
    /// The cursors created by given factories are then used to walk through the accounts and
    /// calculate the state root value with.
    pub fn new(trie_cursor_factory: T, hashed_cursor_factory: H) -> Self {
        Self {
            trie_cursor_factory,
            hashed_cursor_factory,
            prefix_sets: TriePrefixSets::default(),
            previous_state: None,
            threshold: 100_000,
            #[cfg(feature = "metrics")]
            metrics: StateRootMetrics::default(),
        }
    }

    /// Set the prefix sets.
    pub fn with_prefix_sets(mut self, prefix_sets: TriePrefixSets) -> Self {
        self.prefix_sets = prefix_sets;
        self
    }

    /// Set the threshold.
    pub const fn with_threshold(mut self, threshold: u64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the threshold to maximum value so that intermediate progress is not returned.
    pub const fn with_no_threshold(mut self) -> Self {
        self.threshold = u64::MAX;
        self
    }

    /// Set the previously recorded intermediate state.
    pub fn with_intermediate_state(mut self, state: Option<IntermediateStateRootState>) -> Self {
        self.previous_state = state;
        self
    }

    /// Set the hashed cursor factory.
    pub fn with_hashed_cursor_factory<HF>(self, hashed_cursor_factory: HF) -> StateRoot<T, HF> {
        StateRoot {
            trie_cursor_factory: self.trie_cursor_factory,
            hashed_cursor_factory,
            prefix_sets: self.prefix_sets,
            threshold: self.threshold,
            previous_state: self.previous_state,
            #[cfg(feature = "metrics")]
            metrics: self.metrics,
        }
    }

    /// Set the trie cursor factory.
    pub fn with_trie_cursor_factory<TF>(self, trie_cursor_factory: TF) -> StateRoot<TF, H> {
        StateRoot {
            trie_cursor_factory,
            hashed_cursor_factory: self.hashed_cursor_factory,
            prefix_sets: self.prefix_sets,
            threshold: self.threshold,
            previous_state: self.previous_state,
            #[cfg(feature = "metrics")]
            metrics: self.metrics,
        }
    }
}

impl<T, H> StateRoot<T, H>
where
    T: TrieCursorFactory + Clone,
    H: HashedCursorFactory + Clone,
{
    /// Walks the intermediate nodes of existing state trie (if any) and hashed entries. Feeds the
    /// nodes into the hash builder. Collects the updates in the process.
    ///
    /// Ignores the threshold.
    ///
    /// # Returns
    ///
    /// The intermediate progress of state root computation and the trie updates.
    pub fn root_with_updates(self) -> Result<(B256, TrieUpdates), StateRootError> {
        match self.with_no_threshold().calculate(true)? {
            StateRootProgress::Complete(root, _, updates) => Ok((root, updates)),
            StateRootProgress::Progress(..) => unreachable!(), // unreachable threshold
        }
    }

    /// Walks the intermediate nodes of existing state trie (if any) and hashed entries. Feeds the
    /// nodes into the hash builder.
    ///
    /// # Returns
    ///
    /// The state root hash.
    pub fn root(self) -> Result<B256, StateRootError> {
        match self.calculate(false)? {
            StateRootProgress::Complete(root, _, _) => Ok(root),
            StateRootProgress::Progress(..) => unreachable!(), // update retenion is disabled
        }
    }

    /// Walks the intermediate nodes of existing state trie (if any) and hashed entries. Feeds the
    /// nodes into the hash builder. Collects the updates in the process.
    ///
    /// # Returns
    ///
    /// The intermediate progress of state root computation.
    pub fn root_with_progress(self) -> Result<StateRootProgress, StateRootError> {
        self.calculate(true)
    }

    fn calculate(self, retain_updates: bool) -> Result<StateRootProgress, StateRootError> {
        trace!(target: "trie::state_root", "calculating state root");
        Ok(StateRootProgress::Complete(B256::default(), usize::default(), TrieUpdates::default()))
    }
}

/// `StorageRoot` is used to compute the root node of an account storage trie.
#[derive(Debug)]
pub struct StorageRoot<T, H> {
    /// A reference to the database transaction.
    pub trie_cursor_factory: T,
    /// The factory for hashed cursors.
    pub hashed_cursor_factory: H,
    /// The hashed address of an account.
    pub hashed_address: B256,
    /// The set of storage slot prefixes that have changed.
    pub prefix_set: PrefixSet,
    /// Storage root metrics.
    #[cfg(feature = "metrics")]
    metrics: TrieRootMetrics,
}

impl<T, H> StorageRoot<T, H> {
    /// Creates a new storage root calculator given a raw address.
    pub fn new(
        trie_cursor_factory: T,
        hashed_cursor_factory: H,
        address: Address,
        prefix_set: PrefixSet,
        #[cfg(feature = "metrics")] metrics: TrieRootMetrics,
    ) -> Self {
        Self::new_hashed(
            trie_cursor_factory,
            hashed_cursor_factory,
            keccak256(address),
            prefix_set,
            #[cfg(feature = "metrics")]
            metrics,
        )
    }

    /// Creates a new storage root calculator given a hashed address.
    pub const fn new_hashed(
        trie_cursor_factory: T,
        hashed_cursor_factory: H,
        hashed_address: B256,
        prefix_set: PrefixSet,
        #[cfg(feature = "metrics")] metrics: TrieRootMetrics,
    ) -> Self {
        Self {
            trie_cursor_factory,
            hashed_cursor_factory,
            hashed_address,
            prefix_set,
            #[cfg(feature = "metrics")]
            metrics,
        }
    }

    /// Set the changed prefixes.
    pub fn with_prefix_set(mut self, prefix_set: PrefixSet) -> Self {
        self.prefix_set = prefix_set;
        self
    }

    /// Set the hashed cursor factory.
    pub fn with_hashed_cursor_factory<HF>(self, hashed_cursor_factory: HF) -> StorageRoot<T, HF> {
        StorageRoot {
            trie_cursor_factory: self.trie_cursor_factory,
            hashed_cursor_factory,
            hashed_address: self.hashed_address,
            prefix_set: self.prefix_set,
            #[cfg(feature = "metrics")]
            metrics: self.metrics,
        }
    }

    /// Set the trie cursor factory.
    pub fn with_trie_cursor_factory<TF>(self, trie_cursor_factory: TF) -> StorageRoot<TF, H> {
        StorageRoot {
            trie_cursor_factory,
            hashed_cursor_factory: self.hashed_cursor_factory,
            hashed_address: self.hashed_address,
            prefix_set: self.prefix_set,
            #[cfg(feature = "metrics")]
            metrics: self.metrics,
        }
    }
}

impl<T, H> StorageRoot<T, H>
where
    T: TrieCursorFactory,
    H: HashedCursorFactory,
{
    /// Walks the hashed storage table entries for a given address and calculates the storage root.
    ///
    /// # Returns
    ///
    /// The storage root and storage trie updates for a given address.
    pub fn root_with_updates(self) -> Result<(B256, usize, StorageTrieUpdates), StorageRootError> {
        self.calculate(true)
    }

    /// Walks the hashed storage table entries for a given address and calculates the storage root.
    ///
    /// # Returns
    ///
    /// The storage root.
    pub fn root(self) -> Result<B256, StorageRootError> {
        let (root, _, _) = self.calculate(false)?;
        Ok(root)
    }

    /// Walks the hashed storage table entries for a given address and calculates the storage root.
    ///
    /// # Returns
    ///
    /// The storage root, number of walked entries and trie updates
    /// for a given address if requested.
    pub fn calculate(self, _: bool) -> Result<(B256, usize, StorageTrieUpdates), StorageRootError> {
        //TODO: Make a storage root call and thing will work fine
        trace!(target: "trie::storage_root", hashed_address = ?self.hashed_address, "calculating storage root");
        Ok((B256::default(), usize::default(), StorageTrieUpdates::default()))
    }
}
