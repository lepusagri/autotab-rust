//! Opt-in checks against the actual model host and its public weight CDN.
//! HEAD requests validate TLS and redirects without downloading model weights.
#[test]
#[ignore = "Requires internet access and the machine's trusted certificate store"]
fn hugging_face_tls_and_weight_redirects() {
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(45))
        .build()
        .unwrap();
    for url in [
        "https://huggingface.co/MuScriptor/muscriptor-large/resolve/8809fdfbed2affa7ade94a7059e746e3880720e7/model.safetensors",
        "https://huggingface.co/gpt2/resolve/main/model.safetensors",
    ] {
        let mut request = client.head(url);
        if url.contains("/MuScriptor/")
            && let Ok(token) = std::env::var("HF_TOKEN")
        {
            request = request.bearer_auth(token);
        }
        let response = request
            .send()
            .expect("HTTPS certificate validation and redirect handling must succeed");
        let status = response.status();
        // A gated model can deny access after a successful TLS handshake.
        assert!(
            status.is_success() || matches!(status.as_u16(), 401 | 403),
            "Unexpected HTTP status: {status}"
        );
        println!(
            "Verified TLS: {} — HTTP {status}",
            response.url().host_str().unwrap()
        );
        if status.as_u16() == 403
            && let Some(message) = response
                .headers()
                .get("x-error-message")
                .and_then(|v| v.to_str().ok())
        {
            let message = if let Ok(token) = std::env::var("HF_TOKEN") {
                if token.is_empty() {
                    message.to_owned()
                } else {
                    message.replace(&token, "[redacted]")
                }
            } else {
                message.to_owned()
            };
            println!("Model access: {message}");
        }
    }
}
