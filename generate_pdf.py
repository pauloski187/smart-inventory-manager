#!/usr/bin/env python3
"""
Generate PDF from markdown documentation.
Uses markdown to HTML conversion with inline CSS for styling.
"""

import markdown
from pathlib import Path

def md_to_html(md_content: str) -> str:
    """Convert markdown to HTML with styling."""

    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'toc']
    )

    # Create styled HTML document
    styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Smart Inventory Manager Documentation</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 210mm;
            margin: 0 auto;
            padding: 20px;
            font-size: 11pt;
        }}

        h1 {{
            color: #2563eb;
            border-bottom: 3px solid #2563eb;
            padding-bottom: 10px;
            page-break-after: avoid;
        }}

        h2 {{
            color: #1e40af;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
            margin-top: 30px;
            page-break-after: avoid;
        }}

        h3 {{
            color: #1e3a8a;
            margin-top: 20px;
            page-break-after: avoid;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 10pt;
        }}

        th, td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }}

        th {{
            background-color: #2563eb;
            color: white;
            font-weight: bold;
        }}

        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}

        tr:hover {{
            background-color: #e0f2fe;
        }}

        code {{
            background-color: #f1f5f9;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 10pt;
        }}

        pre {{
            background-color: #1e293b;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 9pt;
            line-height: 1.4;
        }}

        pre code {{
            background-color: transparent;
            padding: 0;
            color: inherit;
        }}

        blockquote {{
            border-left: 4px solid #2563eb;
            margin: 15px 0;
            padding: 10px 20px;
            background-color: #eff6ff;
            font-style: italic;
        }}

        hr {{
            border: none;
            border-top: 2px solid #e2e8f0;
            margin: 30px 0;
        }}

        .success {{
            color: #16a34a;
        }}

        .warning {{
            color: #ea580c;
        }}

        .error {{
            color: #dc2626;
        }}

        strong {{
            color: #1e40af;
        }}

        em {{
            color: #64748b;
        }}

        /* Page break controls */
        h1, h2 {{
            page-break-after: avoid;
        }}

        table, pre {{
            page-break-inside: avoid;
        }}

        /* Cover page styling */
        .cover {{
            text-align: center;
            padding: 100px 0;
        }}

        .cover h1 {{
            font-size: 28pt;
            border: none;
        }}

        .cover p {{
            font-size: 14pt;
            color: #64748b;
        }}
    </style>
</head>
<body>
    {html_content}

    <hr>
    <p style="text-align: center; color: #64748b; font-size: 10pt;">
        <strong>Smart Inventory Manager</strong> | Version 2.1.0 | December 2024<br>
        Repository: github.com/pauloski187/smart-inventory-manager
    </p>
</body>
</html>
"""
    return styled_html


def main():
    # Read markdown file
    md_path = Path(__file__).parent / 'SMART_INVENTORY_MANAGER_DOCUMENTATION.md'

    if not md_path.exists():
        print(f"Error: {md_path} not found")
        return

    md_content = md_path.read_text(encoding='utf-8')

    # Convert to HTML
    html_content = md_to_html(md_content)

    # Save HTML (can be opened in browser and printed to PDF)
    html_path = Path(__file__).parent / 'SMART_INVENTORY_MANAGER_DOCUMENTATION.html'
    html_path.write_text(html_content, encoding='utf-8')

    print(f"✓ HTML documentation generated: {html_path}")
    print(f"\nTo create PDF:")
    print(f"  1. Open the HTML file in your browser")
    print(f"  2. Press Ctrl+P (or Cmd+P on Mac)")
    print(f"  3. Select 'Save as PDF' as the destination")
    print(f"  4. Save as 'SMART_INVENTORY_MANAGER_DOCUMENTATION.pdf'")


if __name__ == '__main__':
    main()
