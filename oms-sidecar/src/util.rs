/// Truncate a string at a UTF-8 codepoint boundary at or below `max_bytes`.
/// Slicing a `&str` at an arbitrary byte index panics if the index lands in
/// the middle of a multi-byte sequence; this helper walks back to the
/// previous boundary instead.
pub fn truncate_safely(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut idx = max_bytes;
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    &s[..idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_under_limit() {
        assert_eq!(truncate_safely("hello", 10), "hello");
    }

    #[test]
    fn ascii_at_limit() {
        assert_eq!(truncate_safely("hello", 5), "hello");
    }

    #[test]
    fn ascii_over_limit() {
        assert_eq!(truncate_safely("hello world", 5), "hello");
    }

    #[test]
    fn multibyte_does_not_panic() {
        // "héllo" — 'é' is 2 bytes (0xC3 0xA9). Truncating at 2 bytes lands
        // mid-codepoint, so we should walk back to byte 1.
        let s = "héllo";
        let out = truncate_safely(s, 2);
        assert_eq!(out, "h");
    }

    #[test]
    fn emoji_does_not_panic() {
        // "🚀abc" — '🚀' is 4 bytes. Truncating at 3 walks back to 0.
        let s = "🚀abc";
        let out = truncate_safely(s, 3);
        assert_eq!(out, "");
    }

    #[test]
    fn empty_string() {
        assert_eq!(truncate_safely("", 5), "");
    }
}
