use anyhow::{Context, Result, ensure};
use rusty_ytdl::{VideoOptions, VideoQuality, VideoSearchOptions, blocking::Video};
use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Duration,
};

pub fn audio(url: &str, work: &Path) -> Result<PathBuf> {
    let parsed = reqwest::Url::parse(url).context("Invalid audio URL")?;
    ensure!(
        matches!(parsed.scheme(), "http" | "https"),
        "Only HTTP(S) URLs are supported"
    );
    let output = work.join("download.audio");
    let host = parsed.host_str().unwrap_or_default();
    if host == "youtu.be" || host == "youtube.com" || host.ends_with(".youtube.com") {
        let options = VideoOptions {
            quality: VideoQuality::HighestAudio,
            filter: VideoSearchOptions::Audio,
            ..Default::default()
        };
        let video = Video::new_with_options(url, options)
            .context("Cannot initialize YouTube downloader")?;
        video.download(&output).context("YouTube download failed. Public videos may also be blocked by YouTube; download the audio separately and use --input.")?;
    } else {
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(20))
            .timeout(Duration::from_secs(3600))
            .build()?;
        let mut response = client.get(parsed).send()?.error_for_status()?;
        let mut file = File::create(&output)?;
        response
            .copy_to(&mut file)
            .context("Audio download failed")?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };

    #[test]
    fn direct_http_download_preserves_bytes() {
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/audio.wav", server.local_addr().unwrap());
        let worker = thread::spawn(move || {
            let (mut stream, _) = server.accept().unwrap();
            let mut request = [0_u8; 4096];
            let count = stream.read(&mut request).unwrap();
            assert!(count > 0);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\nRIFFtest",
                )
                .unwrap();
        });
        let work = tempfile::tempdir().unwrap();
        let file = audio(&url, work.path()).unwrap();
        assert_eq!(std::fs::read(file).unwrap(), b"RIFFtest");
        worker.join().unwrap();
    }

    #[test]
    fn rejects_non_http_urls() {
        let work = tempfile::tempdir().unwrap();
        assert!(audio("file:///C:/song.mp3", work.path()).is_err());
    }
}
