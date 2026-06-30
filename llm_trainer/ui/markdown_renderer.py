from __future__ import annotations

import re
from urllib.parse import quote


def markdown_to_html(markdown_text: str) -> str:
    """Convert Markdown to themed HTML.

    Args:
        markdown_text: Markdown content.

    Returns:
        HTML suitable for a chat bubble.
    """

    try:
        import markdown as markdown_lib
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import TextLexer, get_lexer_by_name, guess_lexer

        markdown_text = normalize_code_blocks(markdown_text)
        body_parts: list[str] = []
        pattern = re.compile(r"```(?P<lang>[\w+-]*)\n(?P<code>.*?)```", re.DOTALL)
        last = 0
        formatter = HtmlFormatter(style="monokai", noclasses=True, nowrap=True)
        for match in pattern.finditer(markdown_text):
            prose = markdown_text[last:match.start()]
            if prose.strip():
                body_parts.append(markdown_lib.markdown(prose, extensions=["tables", "nl2br"]))
            code = match.group("code")
            lang = match.group("lang").strip()
            try:
                lexer = get_lexer_by_name(lang) if lang else guess_lexer(code)
            except Exception:
                lexer = TextLexer()
            highlighted = highlight(code, lexer, formatter)
            label = lang.title() if lang else lexer.name
            body_parts.append(code_block_html(label, highlighted, code))
            last = match.end()
        prose = markdown_text[last:]
        if prose.strip():
            body_parts.append(markdown_lib.markdown(prose, extensions=["tables", "nl2br"]))
        body = "\n".join(body_parts) if body_parts else ""
        return (
            f"""<!doctype html>
            <html>
            <head>
            <style>
            body {{
                background: transparent;
                color: #eeeeee;
                font-family: Arial, "Segoe UI", sans-serif;
                font-size: 14px;
                line-height: 1.22;
                margin: 0;
            }}
            h1 {{ color: #f2f2f2; font-size: 19px; margin: 6px 0 3px 0; }}
            h2 {{ color: #f2f2f2; font-size: 17px; margin: 6px 0 3px 0; }}
            h3 {{ color: #f2f2f2; font-size: 15px; margin: 5px 0 2px 0; }}
            p {{ margin: 2px 0; }}
            ol, ul {{ margin-top: 2px; margin-bottom: 2px; padding-left: 20px; }}
            li {{ margin: 1px 0; }}
            code {{
                background: #1a1a1a;
                color: #d4d4d4;
                border-radius: 4px;
                padding: 2px 4px;
                font-family: Consolas, monospace;
            }}
            pre {{
                background: transparent;
                border: 0;
                border-radius: 0;
                padding: 0;
                margin: 4px 0;
                overflow: auto;
                white-space: pre-wrap;
            }}
            pre code {{
                background: transparent;
                padding: 0;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                font-size: 13px;
                line-height: 1.16;
            }}
            blockquote {{
                border-left: 3px solid #f5b041;
                margin-left: 0;
                padding-left: 12px;
                color: #cccccc;
            }}
            table {{ border-collapse: collapse; }}
            th, td {{ border: 1px solid #555555; padding: 6px 8px; }}
            .codeblock {{
                background: #050505;
                border: 1px solid #2b2b2b;
                border-radius: 12px;
                margin: 8px 0;
            }}
            .codebar {{
                color: #f2f2f2;
                background: #111111;
                border-bottom: 1px solid #2b2b2b;
                padding: 7px 10px;
                font-size: 12px;
                font-weight: bold;
            }}
            .copylink {{
                color: #d7d7d7;
                text-decoration: none;
                float: right;
                font-weight: normal;
            }}
            .codebody {{ padding: 12px 14px; }}
            </style>
            </head>
            <body>{body}</body>
            </html>
            """
        )
    except Exception:
        return basic_markdown_html(markdown_text)

def basic_markdown_html(markdown_text: str) -> str:
    """Render basic Markdown with simple code coloring.

    Args:
        markdown_text: Raw Markdown.

    Returns:
        Basic HTML.
    """

    text = normalize_code_blocks(markdown_text)
    parts: list[str] = []
    pattern = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
    last = 0
    for match in pattern.finditer(text):
        parts.append(render_basic_prose(text[last:match.start()]))
        code = match.group(1)
        parts.append(code_block_html("Code", colorize_code(code), code))
        last = match.end()
    parts.append(render_basic_prose(text[last:]))
    return (
        "<html><body style='background:transparent;color:#eeeeee;font-family:Arial;font-size:14px;line-height:1.22;'>"
        "<style>"
        "p{margin:2px 0;} h1{font-size:19px;margin:6px 0 3px;} h2{font-size:17px;margin:6px 0 3px;}"
        "h3{font-size:15px;margin:5px 0 2px;} ol,ul{margin-top:2px;margin-bottom:2px;padding-left:20px;} li{margin:1px 0;}"
        "code{background:#1a1a1a;color:#d4d4d4;border-radius:4px;padding:2px 4px;font-family:Consolas,monospace;}"
        "pre{background:transparent;border:0;border-radius:0;padding:0;margin:4px 0;"
        "font-family:Consolas,monospace;white-space:pre-wrap;line-height:1.16;font-size:13px;}"
        ".codeblock{background:#050505;border:1px solid #2b2b2b;border-radius:12px;margin:8px 0;}"
        ".codebar{color:#f2f2f2;background:#111;border-bottom:1px solid #2b2b2b;padding:7px 10px;font-size:12px;font-weight:bold;}"
        ".copylink{color:#d7d7d7;text-decoration:none;float:right;font-weight:normal;}.codebody{padding:12px 14px;}"
        "</style>"
        + "".join(parts)
        + "</body></html>"
    )

def code_block_html(label: str, highlighted_html: str, raw_code: str) -> str:
    """Build a code panel with a copy link.

    Args:
        label: Code language label.
        highlighted_html: Highlighted code HTML.
        raw_code: Raw code for clipboard copy.

    Returns:
        Code panel HTML.
    """

    return (
        "<div class='codeblock'>"
        f"<div class='codebar'>{escape_html(label or 'Code')}"
        f"<a class='copylink' href='copycode:{quote(raw_code)}'>⧉ Copy</a></div>"
        f"<div class='codebody'><pre><code>{highlighted_html}</code></pre></div>"
        "</div>"
    )

def render_basic_prose(text: str) -> str:
    """Render a small Markdown subset for fallback mode.

    Args:
        text: Markdown prose.

    Returns:
        HTML fragment.
    """

    html_lines: list[str] = []
    in_ordered = False
    in_unordered = False

    def close_lists() -> None:
        nonlocal in_ordered, in_unordered
        if in_ordered:
            html_lines.append("</ol>")
            in_ordered = False
        if in_unordered:
            html_lines.append("</ul>")
            in_unordered = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            close_lists()
            html_lines.append("<br>")
            continue
        if line.startswith("### "):
            close_lists()
            html_lines.append(f"<h3>{inline_basic_markdown(line[4:])}</h3>")
            continue
        if line.startswith("## "):
            close_lists()
            html_lines.append(f"<h2>{inline_basic_markdown(line[3:])}</h2>")
            continue
        if line.startswith("# "):
            close_lists()
            html_lines.append(f"<h1>{inline_basic_markdown(line[2:])}</h1>")
            continue
        ordered = re.match(r"^\d+\.\s+(.*)$", line)
        if ordered:
            if not in_ordered:
                close_lists()
                html_lines.append("<ol>")
                in_ordered = True
            html_lines.append(f"<li>{inline_basic_markdown(ordered.group(1))}</li>")
            continue
        unordered = re.match(r"^[-*]\s+(.*)$", line)
        if unordered:
            if not in_unordered:
                close_lists()
                html_lines.append("<ul>")
                in_unordered = True
            html_lines.append(f"<li>{inline_basic_markdown(unordered.group(1))}</li>")
            continue
        close_lists()
        html_lines.append(f"<p>{inline_basic_markdown(line)}</p>")
    close_lists()
    return "\n".join(html_lines)

def inline_basic_markdown(text: str) -> str:
    """Render inline Markdown for fallback mode.

    Args:
        text: Inline Markdown text.

    Returns:
        HTML fragment.
    """

    escaped = escape_html(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*(.+?)\*", r"<em>\1</em>", escaped)
    return escaped

def escape_html(text: str) -> str:
    """Escape text for HTML.

    Args:
        text: Raw text.

    Returns:
        Escaped text.
    """

    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def colorize_code(code: str) -> str:
    """Apply simple inline colors to Python-like code.

    Args:
        code: Source code.

    Returns:
        HTML code.
    """

    escaped = escape_html(code)
    keywords = {
        "def", "class", "import", "from", "for", "while", "if", "else", "elif",
        "try", "except", "return", "print", "with", "as", "in", "function", "const",
        "let", "var", "new", "typeof", "await", "async", "true", "false", "null",
        "True", "False", "None",
    }
    builtins = {"console", "Object", "process", "JSON", "Array", "String", "Number", "Boolean", "Math", "os", "sys"}
    token_pattern = re.compile(
        r"(?P<comment>//.*|#.*)"
        r"|(?P<string>`(?:\\.|[^`])*`|'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\")"
        r"|(?P<number>\b\d+(?:\.\d+)?\b)"
        r"|(?P<word>\b[A-Za-z_][A-Za-z0-9_]*\b)"
    )
    colored_lines: list[str] = []
    for line in escaped.splitlines():
        segments: list[str] = []
        last = 0
        for match in token_pattern.finditer(line):
            segments.append(line[last:match.start()])
            value = match.group(0)
            if match.lastgroup == "comment":
                segments.append(f"<span style='color:#6a9955;'>{value}</span>")
            elif match.lastgroup == "string":
                segments.append(f"<span style='color:#ce9178;'>{value}</span>")
            elif match.lastgroup == "number":
                segments.append(f"<span style='color:#b5cea8;'>{value}</span>")
            elif match.lastgroup == "word":
                next_chars = line[match.end(): match.end() + 2]
                previous = line[max(0, match.start() - 1): match.start()]
                if value in keywords:
                    segments.append(f"<span style='color:#569cd6;font-weight:bold;'>{value}</span>")
                elif value in builtins:
                    segments.append(f"<span style='color:#4ec9b0;'>{value}</span>")
                elif next_chars.startswith("(") and previous != ".":
                    segments.append(f"<span style='color:#dcdcaa;'>{value}</span>")
                elif previous == ".":
                    segments.append(f"<span style='color:#9cdcfe;'>{value}</span>")
                else:
                    segments.append(value)
            last = match.end()
        segments.append(line[last:])
        colored_lines.append("".join(segments))
    return "\n".join(colored_lines)

def normalize_code_blocks(markdown_text: str) -> str:
    """Fence obvious loose code blocks so syntax highlighting can run.

    Args:
        markdown_text: Raw model Markdown.

    Returns:
        Markdown with likely code blocks fenced.
    """

    if "```" in markdown_text:
        if markdown_text.count("```") % 2:
            return f"{markdown_text}\n```"
        return markdown_text
    lines = markdown_text.splitlines()
    normalized: list[str] = []
    code_block: list[str] = []

    def is_code_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return bool(code_block)
        if line.startswith(("    ", "\t")):
            return True
        if re.match(
            r"^(def|class|import|from|for|while|if|else:?|elif|try:?|except|return|print|with|"
            r"function|const|let|var|console\.|Object\.|process\.)\b",
            stripped,
        ):
            return True
        if stripped in {"{", "}", "};", "})", "});"}:
            return True
        if stripped.startswith(("#", "@")):
            return True
        return sum(stripped.count(symbol) for symbol in "()[]{}:=<>+-*/") >= 3

    def flush() -> None:
        nonlocal code_block
        if len([line for line in code_block if line.strip()]) >= 3:
            normalized.append(f"```{guess_code_language(code_block)}")
            normalized.extend(code_block)
            normalized.append("```")
        else:
            normalized.extend(code_block)
        code_block = []

    for line in lines:
        if is_code_line(line):
            code_block.append(line)
        else:
            flush()
            normalized.append(line)
    flush()
    return "\n".join(normalized)

def guess_code_language(lines: list[str]) -> str:
    """Guess a fence language for loose code.

    Args:
        lines: Code lines.

    Returns:
        Markdown fence language.
    """

    joined = "\n".join(lines).lower()
    if any(marker in joined for marker in ("console.", "const ", "let ", "function ", "process.env", "object.keys")):
        return "javascript"
    if any(marker in joined for marker in ("#include", "std::", "cout", "cin")):
        return "cpp"
    if any(marker in joined for marker in ("public class", "system.out", "private ", "protected ")):
        return "java"
    if any(marker in joined for marker in ("def ", "import ", "print(", "self.")):
        return "python"
    return "text"

