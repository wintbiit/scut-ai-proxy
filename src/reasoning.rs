pub fn clean_reasoning(raw: &str) -> String {
    let mut output = String::new();
    let mut rest = raw;
    let mut removed_any = false;

    loop {
        let Some(start) = rest.find("<think>") else {
            output.push_str(rest);
            break;
        };
        output.push_str(&rest[..start]);
        removed_any = true;
        let after_start = &rest[start + "<think>".len()..];
        if let Some(end) = after_start.find("</think>") {
            rest = &after_start[end + "</think>".len()..];
        } else {
            break;
        }
    }

    let cleaned = output.replace("</think>", "").trim().to_string();
    if cleaned.is_empty() && !raw.trim().is_empty() {
        tracing::warn!(
            "reasoning cleanup produced empty content; falling back to visible upstream text"
        );
        raw.replace("<think>", "")
            .replace("</think>", "")
            .trim()
            .to_string()
    } else if removed_any {
        cleaned
    } else {
        raw.trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_complete_think_block() {
        assert_eq!(clean_reasoning("<think>hidden</think>答案"), "答案");
    }

    #[test]
    fn handles_only_open_tag() {
        assert_eq!(clean_reasoning("<think>"), "");
    }

    #[test]
    fn keeps_plain_text() {
        assert_eq!(clean_reasoning("答案"), "答案");
    }
}
