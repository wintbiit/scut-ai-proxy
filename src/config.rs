use std::{env, time::Duration};

#[derive(Clone, Debug)]
pub struct Config {
    pub bind_addr: String,
    pub chat3_base_url: String,
    pub request_timeout: Duration,
    pub planner_repair_attempts: usize,
}

impl Config {
    pub fn from_env() -> Self {
        let bind_addr = env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:3000".to_string());
        let chat3_base_url = env::var("CHAT3_BASE_URL")
            .unwrap_or_else(|_| "https://chat3.scut.edu.cn/api".to_string())
            .trim_end_matches('/')
            .to_string();
        let request_timeout_secs = env::var("REQUEST_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(120);
        let planner_repair_attempts = env::var("PLANNER_REPAIR_ATTEMPTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1);

        Self {
            bind_addr,
            chat3_base_url,
            request_timeout: Duration::from_secs(request_timeout_secs),
            planner_repair_attempts,
        }
    }
}
