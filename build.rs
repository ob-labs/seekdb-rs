//! Build script for seekdb-rs.
//!
//! When `embedded` feature is enabled, libseekdb is downloaded from OceanBase S3
//! (cached under `target/libseekdb`, reused on subsequent builds).
//!
//! Download URL format:
//! `https://oceanbase-seekdb-builds.s3.ap-southeast-1.amazonaws.com/libseekdb/all_commits/26914ab0c7717c873c4b1d7665c74cf4aac4fab3/libseekdb-{os}-{arch}.zip`

use std::env;
#[cfg(feature = "embedded")]
use std::fs;
#[cfg(feature = "embedded")]
use std::io;
#[cfg(feature = "embedded")]
use std::path::{Path, PathBuf};

#[cfg(feature = "embedded")]
const SEEKDB_DOWNLOAD_BASE: &str =
    "https://oceanbase-seekdb-builds.s3.ap-southeast-1.amazonaws.com/libseekdb/all_commits/26914ab0c7717c873c4b1d7665c74cf4aac4fab3";

fn main() {
    if env::var("CARGO_FEATURE_EMBEDDED").is_err() {
        return;
    }

    #[cfg(feature = "embedded")]
    {
        let out_dir = env::var("OUT_DIR").unwrap();
        let out_path = Path::new(&out_dir).join("bindgen.rs");

        let (lib_dir, opt_header) =
            download_libseekdb(&out_dir).expect("Failed to download libseekdb");
        let header_path = opt_header.expect(
            "Downloaded libseekdb archive must contain seekdb.h (in root or include/).",
        );

        if !header_path.exists() {
            panic!("seekdb.h not found at: {}", header_path.display());
        }

        let bindings = bindgen::Builder::default()
            .header(header_path.to_string_lossy())
            .allowlist_type("Seekdb.*")
            .allowlist_function("seekdb.*")
            .allowlist_var("SEEKDB.*")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate bindings");

        bindings.write_to_file(&out_path).expect("Couldn't write bindings!");

        let lib_dir_str = lib_dir.as_path().to_string_lossy();
        println!("cargo:rustc-link-search=native={}", lib_dir_str);
        println!("cargo:rustc-link-lib=dylib=seekdb");

        #[cfg(target_os = "linux")]
        {
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../");
        }
        #[cfg(target_os = "macos")]
        {
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../");
        }

        for name in ["libseekdb.so", "libseekdb.dylib"] {
            let lib_path = lib_dir.join(name);
            if lib_path.exists() {
                println!("cargo:rerun-if-changed={}", lib_path.display());
            }
        }
        println!("cargo:rerun-if-changed={}", header_path.display());
    }
}

/// (TARGET triple, archive_name, dynamic_lib_filename). Order does not matter.
#[cfg(feature = "embedded")]
const LIBSEEKDB_TARGETS: &[(&str, &str, &str)] = &[
    ("x86_64-unknown-linux-gnu", "libseekdb-linux-x64.zip", "libseekdb.so"),
    ("aarch64-unknown-linux-gnu", "libseekdb-linux-arm64.zip", "libseekdb.so"),
    ("aarch64-apple-darwin", "libseekdb-darwin-arm64.zip", "libseekdb.dylib"),
    // ("x86_64-apple-darwin", "libseekdb-darwin-x86_64.zip", "libseekdb.dylib"), // add when available
];

/// Returns (archive_name, dynamic_lib_filename) for the current TARGET.
#[cfg(feature = "embedded")]
fn libseekdb_archive_for_target(target: &str) -> Option<(&'static str, &'static str)> {
    LIBSEEKDB_TARGETS
        .iter()
        .find(|(t, _, _)| *t == target)
        .map(|(_, archive, lib)| (*archive, *lib))
}

/// Returns (lib_dir, optional header path if seekdb.h found inside extracted archive).
#[cfg(feature = "embedded")]
fn download_libseekdb(out_dir: &str) -> Result<(PathBuf, Option<PathBuf>), Box<dyn std::error::Error>> {
    let target = env::var("TARGET")?;
    let (archive_name, dynamic_lib) = libseekdb_archive_for_target(&target)
        .ok_or_else(|| format!("No pre-built libseekdb available for target '{target}'"))?;

    let download_dir = workspace_download_dir(out_dir)?;
    fs::create_dir_all(&download_dir)?;

    let archive_path = download_dir.join(archive_name);
    let lib_marker = download_dir.join(dynamic_lib);

    if lib_marker.exists() {
        println!("cargo:warning=Reusing libseekdb from {}", download_dir.display());
    } else {
        let client = http_client()?;
        let url = format!("{}/{}", SEEKDB_DOWNLOAD_BASE, archive_name);
        ensure_libseekdb(&client, &url, &archive_path)?;
        extract_archive(&archive_path, &download_dir)?;
        if !lib_marker.exists() {
            return Err(format!(
                "Downloaded archive did not contain expected library '{}'",
                dynamic_lib
            )
            .into());
        }
    }

    copy_lib_to_deps(&download_dir, dynamic_lib, out_dir)?;

    let header_in_extracted = find_seekdb_header_in_dir(&download_dir);
    Ok((download_dir, header_in_extracted))
}

#[cfg(feature = "embedded")]
fn find_seekdb_header_in_dir(dir: &Path) -> Option<PathBuf> {
    let candidates = [dir.join("seekdb.h"), dir.join("include").join("seekdb.h")];
    for p in &candidates {
        if p.exists() {
            return Some(p.clone());
        }
    }
    None
}

#[cfg(feature = "embedded")]
fn workspace_download_dir(out_dir: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Ok(dir) = env::var("CARGO_TARGET_DIR") {
        return Ok(PathBuf::from(dir).join("libseekdb"));
    }
    let target_root = Path::new(out_dir)
        .ancestors()
        .find(|a| {
            a.file_name()
                .and_then(|n| n.to_str())
                .map_or(false, |n| n == "target")
        })
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("target"));
    Ok(target_root.join("libseekdb"))
}

#[cfg(feature = "embedded")]
fn ensure_libseekdb(
    client: &build_reqwest::blocking::Client,
    url: &str,
    archive_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if archive_path.exists() {
        println!("cargo:warning=libseekdb archive already present at {}", archive_path.display());
        return Ok(());
    }
    let tmp_path = archive_path.with_extension("download");
    if let Some(parent) = archive_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut response = client.get(url).send()?;
    let status = response.status();
    if !status.is_success() {
        let hint = if status.as_u16() == 403 {
            " (If the bucket is private, use a pre-built libseekdb from an allowed source.)"
        } else {
            ""
        };
        return Err(format!("HTTP {} for {}{}", status, url, hint).into());
    }
    let mut tmp_file = fs::File::create(&tmp_path)?;
    io::copy(&mut response, &mut tmp_file)?;
    fs::rename(&tmp_path, archive_path)?;
    println!("cargo:warning=Downloaded libseekdb from {}", url);
    Ok(())
}

#[cfg(feature = "embedded")]
fn extract_archive(archive_path: &Path, dest: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let file = fs::File::open(archive_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    archive.extract(dest)?;
    println!("cargo:warning=Extracted libseekdb to {}", dest.display());
    Ok(())
}

#[cfg(feature = "embedded")]
fn profile_deps_dir(out_dir: &str) -> Option<PathBuf> {
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let mut target_root = env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            Path::new(out_dir)
                .ancestors()
                .find(|a| {
                    a.file_name()
                        .and_then(|n| n.to_str())
                        .map_or(false, |n| n == "target")
                })
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("target"))
        });
    if env::var("HOST").ok() != env::var("TARGET").ok() {
        if let Ok(t) = env::var("TARGET") {
            target_root.push(t);
        }
    }
    target_root.push(profile);
    target_root.push("deps");
    Some(target_root)
}

#[cfg(feature = "embedded")]
fn copy_lib_to_deps(
    download_dir: &Path,
    lib_filename: &str,
    out_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(deps_dir) = profile_deps_dir(out_dir) else {
        println!("cargo:warning=Could not determine target/deps directory, skipping runtime copy");
        return Ok(());
    };
    fs::create_dir_all(&deps_dir)?;
    let source = download_dir.join(lib_filename);
    let dest = deps_dir.join(lib_filename);
    if dest.exists() {
        fs::remove_file(&dest)?;
    }
    fs::copy(&source, &dest)?;
    println!("cargo:warning=Copied libseekdb to {}", dest.display());
    Ok(())
}

#[cfg(feature = "embedded")]
fn http_client() -> Result<build_reqwest::blocking::Client, build_reqwest::Error> {
    let timeout = env::var("CARGO_HTTP_TIMEOUT")
        .or_else(|_| env::var("HTTP_TIMEOUT"))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(90);
    build_reqwest::blocking::Client::builder()
        .user_agent("seekdb-rs-build/1.0 (+https://github.com/ob-labs/seekdb-rs)")
        .timeout(std::time::Duration::from_secs(timeout))
        .build()
}
