use reth_storage_errors::db::DatabaseError;
use std::{
    fmt,
    io::{Read, Write},
    os::unix::net::UnixStream,
    result::Result::Ok,
    sync::mpsc::{self, Receiver},
    thread,
    time::Duration,
};

const OP_STATE_ROOT: u8 = 1;
const OP_STATE_PROOF: u8 = 2;

const STATUS_SUCCESS: u8 = 1;
const STATUS_ERROR: u8 = 0;

const SOCKET_PATH: &str = "/tmp/ipc/scalerize_state.sock";
/// Represents errors that can occur while interacting with the Scalerize client.
///
/// This enum is used to categorize different types of errors that may arise during
/// operations such as I/O errors, operation failures, and invalid responses from the server.
#[derive(Debug)]
pub enum ClientError {
    /// An I/O error occurred.
    Io(std::io::Error),

    /// The requested operation failed with a specific message.
    OperationFailed(String),

    /// The request made is invalid.
    InvalidRequest(String),

    /// The response received from the server was invalid.
    InvalidResponse(String),
}

impl From<ClientError> for DatabaseError {
    fn from(error: ClientError) -> Self {
        match error {
            ClientError::Io(err) => DatabaseError::Other(format!("IO error: {}", err)),
            ClientError::InvalidResponse(msg) => {
                DatabaseError::Other(format!("Invalid response: {}", msg))
            }
            ClientError::InvalidRequest(msg) => {
                DatabaseError::Other(format!("Invalid request: {}", msg))
            }
            ClientError::OperationFailed(msg) => {
                DatabaseError::Other(format!("Operation failed: {}", msg))
            }
        }
    }
}

impl fmt::Display for ClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClientError::Io(err) => write!(f, "IO error: {}", err),
            ClientError::InvalidResponse(msg) => write!(f, "Invalid response: {}", msg),
            ClientError::InvalidRequest(msg) => write!(f, "Invalid request: {}", msg),
            ClientError::OperationFailed(msg) => write!(f, "Operation failed: {}", msg),
        }
    }
}

impl From<std::io::Error> for ClientError {
    fn from(error: std::io::Error) -> Self {
        ClientError::Io(error)
    }
}

impl Into<i32> for ClientError {
    fn into(self) -> i32 {
        match self {
            ClientError::Io(_) => 1,
            ClientError::InvalidResponse(_) => 2,
            ClientError::InvalidRequest(_) => 3,
            ClientError::OperationFailed(_) => 4,
        }
    }
}

// Client for making state calls to scalerize
pub struct ScalerizeStateClient {
    stream: UnixStream,
}

impl ScalerizeStateClient {
    pub fn connect() -> Result<Self, ClientError> {
        let stream = UnixStream::connect(SOCKET_PATH)?;
        Ok(Self { stream })
    }

    pub fn spawn_connect_thread() -> Receiver<Result<Self, ClientError>> {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || loop {
            match ScalerizeStateClient::connect() {
                Ok(client) => {
                    let _ = tx.send(Ok(client));
                    break;
                }
                Err(err) => {
                    println!("Failed to connect to state scalerize client: {}. Retrying...", err);
                    thread::sleep(Duration::from_secs(1));
                }
            }
        });
        rx
    }

    fn log_response(response: &[u8]) {
        if response.is_empty() {
            println!("Empty response received");
            return;
        }

        let status = response[0];
        let data = &response[1..];

        println!("Server Response Status: {}", status);
        println!("Raw Response Data: {:?}", data);
        if let Ok(text) = String::from_utf8(data.to_vec()) {
            println!("Response as text: {}", text);
        }
    }

    fn read_full_response(&mut self) -> Result<Vec<u8>, ClientError> {
        let mut response = vec![0u8; 4096];
        let n = self.stream.read(&mut response)?;
        response.truncate(n);

        if response.is_empty() {
            return Err(ClientError::InvalidResponse("Empty response from server".to_string()));
        }

        // Self::log_response(&response);
        Ok(response)
    }

    pub fn state_proof(&mut self, block_spec_bytes: &[u8], serialized_hashed_account_bytes: &[u8], serialized_storage_keys_bytes: &[u8]) -> Result<Option<Vec<u8>>, ClientError> {
        let mut request = vec![OP_STATE_PROOF];
        request.extend_from_slice(block_spec_bytes);
        request.extend_from_slice(serialized_hashed_account_bytes);
        request.extend_from_slice(serialized_storage_keys_bytes);

        self.stream.write_all(&request)?;
        self.stream.flush()?;

        let response = self.read_full_response()?;
        let status = response[0];
        let data = response[1..].to_vec();
        if data.is_empty() {
            return Ok(None)
        }

        match status {
            STATUS_SUCCESS => Ok(Some(data)),
            STATUS_ERROR => {
                Err(ClientError::OperationFailed(String::from_utf8_lossy(&data).into_owned()))
            }
            _ => Err(ClientError::OperationFailed(format!("Error: {:?}", data))),
        }
    }

    pub fn state_root(&mut self, height: &[u8]) -> Result<Option<Vec<u8>>, ClientError> {
        let mut request = vec![OP_STATE_ROOT];
        request.extend_from_slice(height);

        self.stream.write_all(&request)?;
        self.stream.flush()?;

        let response = self.read_full_response()?;
        let status = response[0];
        let data = response[1..].to_vec();
        if data.is_empty() {
            return Ok(None)
        }

        match status {
            STATUS_SUCCESS => Ok(Some(data)),
            STATUS_ERROR => {
                Err(ClientError::OperationFailed(String::from_utf8_lossy(&data).into_owned()))
            }
            _ => Err(ClientError::OperationFailed(format!("Error: {:?}", data))),
        }
    }

    pub fn check_additional_messages(&mut self) {
        println!("Checking for additional messages...");
        // Set socket to non-blocking mode for checking additional messages
        self.stream
            .set_nonblocking(true)
            .unwrap_or_else(|e| println!("Failed to set non-blocking mode: {}", e));

        loop {
            let mut buffer = vec![0u8; 4096];
            match self.stream.read(&mut buffer) {
                Ok(n) if n > 0 => {
                    buffer.truncate(n);
                    println!("Additional message received: {:?}", buffer);
                }
                Ok(_) => {
                    println!("No more messages");
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    println!("No more messages");
                    break;
                }
                Err(e) => {
                    println!("Error reading additional messages: {}", e);
                    break;
                }
            }
        }

        // Set socket back to blocking mode
        self.stream
            .set_nonblocking(false)
            .unwrap_or_else(|e| println!("Failed to set blocking mode: {}", e));
    }
}

impl std::fmt::Debug for ScalerizeStateClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalerizeStateClient")
            .field("stream", &format!("UnixStream connected to {}", SOCKET_PATH))
            .finish()
    }
}
