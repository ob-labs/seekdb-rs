#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(deref_nullptr)]
#![allow(improper_ctypes)]

#[allow(clippy::all)]
#[allow(dead_code)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindgen.rs"));
}

#[allow(clippy::all)]
pub use bindings::*;

// Re-export C API constants (SEEKDB_SUCCESS used in embedded.rs; error codes for API surface).
pub const SEEKDB_SUCCESS: i32 = 0;
#[allow(dead_code)]
pub const SEEKDB_ERROR_INVALID_PARAM: i32 = -1;
#[allow(dead_code)]
pub const SEEKDB_ERROR_CONNECTION_FAILED: i32 = -2;
#[allow(dead_code)]
pub const SEEKDB_ERROR_QUERY_FAILED: i32 = -3;
#[allow(dead_code)]
pub const SEEKDB_ERROR_MEMORY_ALLOC: i32 = -4;
#[allow(dead_code)]
pub const SEEKDB_ERROR_NOT_INITIALIZED: i32 = -5;
