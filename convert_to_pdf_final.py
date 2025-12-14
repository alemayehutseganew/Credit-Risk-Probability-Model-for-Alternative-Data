import markdown
from xhtml2pdf import pisa
import os

def convert_md_to_pdf(source_md_file, output_pdf_file):
    source_dir = os.path.dirname(os.path.abspath(source_md_file))
    
    with open(source_md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

    def link_callback(uri, rel):
        if not uri.startswith('http') and not os.path.isabs(uri):
            path = os.path.join(source_dir, uri)
            return path
        return uri

    with open(output_pdf_file, "wb") as result_file:
        pisa_status = pisa.CreatePDF(
            html_content,
            dest=result_file,
            link_callback=link_callback
        )

    if pisa_status.err:
        print(f"Error converting {source_md_file} to PDF")
    else:
        print(f"Successfully created {output_pdf_file}")

if __name__ == "__main__":
    source = "reports/final_submission_report.md"
    output = "reports/final_submission_report.pdf"
    convert_md_to_pdf(source, output)
