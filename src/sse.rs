use bytes::Bytes;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub data: String,
}

impl SseEvent {
    pub fn is_done(&self) -> bool {
        self.data.trim() == "[DONE]"
    }
}

#[derive(Default)]
pub struct SseParser {
    buffer: String,
}

impl SseParser {
    pub fn push(&mut self, chunk: &Bytes) -> Vec<SseEvent> {
        self.buffer.push_str(&String::from_utf8_lossy(chunk));
        let mut events = Vec::new();

        while let Some(idx) = find_event_boundary(&self.buffer) {
            let raw = self.buffer[..idx].to_string();
            let drain_to = if self.buffer[idx..].starts_with("\r\n\r\n") {
                idx + 4
            } else {
                idx + 2
            };
            self.buffer.drain(..drain_to);
            if let Some(event) = parse_event(&raw) {
                events.push(event);
            }
        }

        events
    }

    pub fn finish(&mut self) -> Vec<SseEvent> {
        if self.buffer.trim().is_empty() {
            self.buffer.clear();
            return Vec::new();
        }
        let raw = std::mem::take(&mut self.buffer);
        parse_event(&raw).into_iter().collect()
    }
}

fn find_event_boundary(buffer: &str) -> Option<usize> {
    let lf = buffer.find("\n\n");
    let crlf = buffer.find("\r\n\r\n");
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn parse_event(raw: &str) -> Option<SseEvent> {
    let data = raw
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("\n");
    if data.is_empty() {
        None
    } else {
        Some(SseEvent { data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_data_events() {
        let mut parser = SseParser::default();
        let events = parser.push(&Bytes::from_static(b"data: {\"a\":1}\n\ndata: [DONE]\n\n"));
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "{\"a\":1}");
        assert!(events[1].is_done());
    }

    #[test]
    fn handles_split_chunks() {
        let mut parser = SseParser::default();
        assert!(parser.push(&Bytes::from_static(b"data: {\"a\"")).is_empty());
        let events = parser.push(&Bytes::from_static(b":1}\n\n"));
        assert_eq!(events[0].data, "{\"a\":1}");
    }
}
