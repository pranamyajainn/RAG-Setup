from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Frame

def generate_pdf(output_filename, title, sections):
    """
    Generate a styled PDF similar to the provided NASDAQ report.
    
    Args:
        output_filename (str): Name of the output PDF file.
        title (str): Title of the report.
        sections (list of dict): Each section has 'heading' and 'content'.
    """
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(name="Title", fontSize=24, leading=30, alignment=1, textColor=colors.HexColor("#003366")))
    styles.add(ParagraphStyle(name="Heading", fontSize=18, leading=22, spaceAfter=10, textColor=colors.HexColor("#003366")))
    styles.add(ParagraphStyle(name="Body", fontSize=12, leading=16, textColor=colors.black))
    
    # Initialize the PDF canvas
    c = canvas.Canvas(output_filename, pagesize=letter)
    width, height = letter
    
    # Title section
    c.setFillColor(colors.HexColor("#003366"))
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width / 2, height - 50, title)
    
    y_position = height - 100  # Start below the title
    
    for section in sections:
        if y_position < 100:  # Add a new page if near bottom
            c.showPage()
            y_position = height - 50
        
        # Add Heading
        heading = Paragraph(section["heading"], styles["Heading"])
        w, h = heading.wrap(width - 100, height)  # Width, height
        heading.drawOn(c, 50, y_position - h)
        y_position -= h + 10  # Space after heading
        
        # Add Content
        body = Paragraph(section["content"], styles["Body"])
        w, h = body.wrap(width - 100, y_position)
        if y_position - h < 50:  # Check if content fits
            c.showPage()
            y_position = height - 50
        body.drawOn(c, 50, y_position - h)
        y_position -= h + 20  # Space after content
    
    # Save the PDF
    c.save()

# Example Usage
sections = [
    {"heading": "Chairman’s Message", "content": "In 2023, we took steps to fortify our business and ..."},
    {"heading": "Operational Review", "content": "Technology plays a vital role in our industry as it continues to redefine ..."},
    # Add more sections as needed
]

generate_pdf("Annual_Report_Styled.pdf", "NASDAQ Annual Report 2023", sections)


from fpdf import FPDF

class NYSEReportPDF(FPDF):
    def header(self):
        # Add header logo or text
        self.set_font('Arial', 'B', 12)
        self.set_text_color(0, 51, 102)  # NYSE-themed blue
        self.cell(0, 10, 'Callon Petroleum Company - Annual Report 2023', ln=1, align='C')
        self.ln(5)  # Spacing after header

    def footer(self):
        # Add a page number footer
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(169, 169, 169)  # Gray text for footer
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def chapter_title(self, title):
        # Add a chapter title
        self.set_font('Arial', 'B', 16)
        self.set_text_color(0, 51, 102)  # NYSE-themed blue
        self.cell(0, 10, title, ln=1, align='L')
        self.ln(5)  # Spacing after title

    def chapter_body(self, body):
        # Add the main text body
        self.set_font('Arial', '', 12)
        self.set_text_color(0)  # Black text
        self.multi_cell(0, 10, body)
        self.ln()

def generate_nyse_pdf(filename, title, sections):
    """
    Generate a NYSE-styled annual report PDF.
    
    Args:
        filename (str): The output file name for the PDF.
        title (str): The title of the report.
        sections (list of dict): Each section has 'heading' and 'content'.
    """
    pdf = NYSEReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Add report title
    pdf.set_font('Arial', 'B', 20)
    pdf.set_text_color(0, 51, 102)  # NYSE-themed blue
    pdf.cell(0, 10, title, ln=1, align='C')
    pdf.ln(10)

    # Add content sections
    for section in sections:
        pdf.chapter_title(section["heading"])
        pdf.chapter_body(section["content"])

    # Save the PDF
    pdf.output(filename)

# Example usage
sections = [
    {"heading": "Business Overview", "content": "Callon Petroleum Company is focused on the acquisition, exploration, and development of high-quality assets..."},
    {"heading": "Financial Highlights", "content": "In 2023, the company reduced its long-term debt by 14% and initiated a share repurchase program..."},
    # Add more sections here
]

generate_nyse_pdf("NYSE_Styled_Report.pdf", "NYSE Annual Report 2023", sections)

from fpdf import FPDF

class JEEMainReportPDF(FPDF):
    def header(self):
        # Add a header with title
        self.set_font('Arial', 'B', 14)
        self.set_text_color(0, 102, 204)  # Blue theme
        self.cell(0, 10, 'JEE Main Sample Analysis Report', ln=1, align='C')
        self.ln(5)

    def footer(self):
        # Add footer with page numbers
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)  # Gray text
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def section_title(self, title):
        # Format the section titles
        self.set_font('Arial', 'B', 12)
        self.set_text_color(0, 102, 204)  # Blue theme
        self.cell(0, 10, title, ln=1, align='L')
        self.ln(5)

    def section_content(self, content):
        # Format the section content
        self.set_font('Arial', '', 10)
        self.set_text_color(0)  # Black text
        self.multi_cell(0, 8, content)
        self.ln()

    def add_table(self, headers, data):
        # Add a table with headers and rows
        self.set_font('Arial', 'B', 10)
        self.set_fill_color(200, 220, 255)  # Light blue for headers
        self.set_text_color(0)
        col_widths = [40, 50, 50, 50]  # Adjust based on column needs

        # Print table headers
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 10, header, 1, 0, 'C', fill=True)
        self.ln()

        # Print table rows
        self.set_font('Arial', '', 10)
        for row in data:
            for i, item in enumerate(row):
                self.cell(col_widths[i], 10, str(item), 1, 0, 'C')
            self.ln()

def generate_jee_pdf(filename, title, sections, tables=None):
    """
    Generate a JEE-styled analysis report PDF.
    
    Args:
        filename (str): Output file name for the PDF.
        title (str): Title of the report.
        sections (list of dict): Each section has 'heading' and 'content'.
        tables (list of dict): Optional; each table has 'headers' and 'rows'.
    """
    pdf = JEEMainReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Add the main title
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 10, title, ln=1, align='C')
    pdf.ln(10)

    # Add content sections
    for section in sections:
        pdf.section_title(section['heading'])
        pdf.section_content(section['content'])

    # Add tables if any
    if tables:
        for table in tables:
            pdf.add_table(table['headers'], table['rows'])

    # Save the PDF
    pdf.output(filename)

# Example usage
sections = [
    {"heading": "Introduction", "content": "This report provides a comprehensive analysis of JEE Main performance..."},
    {"heading": "Key Insights", "content": "The overall performance indicates an increase in scores for most participants..."},
]

tables = [
    {
        "headers": ["Category", "Total Students", "Average Score", "Top Score"],
        "rows": [
            ["General", 1000, 180, 300],
            ["OBC", 800, 170, 290],
            ["SC", 500, 150, 270],
            ["ST", 300, 140, 260],
        ]
    }
]

generate_jee_pdf("JEE_Main_Analysis_Report.pdf", "JEE Main Sample Analysis Report", sections, tables)


from fpdf import FPDF

class ResearchReportPDF(FPDF):
    def header(self):
        # Add header with title
        self.set_font('Arial', 'B', 12)
        self.set_text_color(0, 51, 102)  # Academic theme blue
        self.cell(0, 10, 'Writing a Research Report - Learning Guide', ln=1, align='C')
        self.ln(5)

    def footer(self):
        # Add footer with page numbers
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)  # Gray footer
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def chapter_title(self, title):
        # Format chapter titles
        self.set_font('Arial', 'B', 14)
        self.set_text_color(0, 51, 102)  # Academic theme blue
        self.cell(0, 10, title, ln=1, align='L')
        self.ln(5)

    def chapter_body(self, body):
        # Format chapter body
        self.set_font('Arial', '', 10)
        self.set_text_color(0)  # Black text
        self.multi_cell(0, 8, body)
        self.ln()

    def add_list(self, items):
        # Add a bulleted list
        self.set_font('Arial', '', 10)
        for item in items:
            self.cell(5)  # Indentation
            self.cell(0, 8, f'• {item}', ln=1)
        self.ln()

def generate_research_guide_pdf(filename, title, chapters):
    """
    Generate a research guide styled PDF.
    
    Args:
        filename (str): The output file name.
        title (str): The title of the report.
        chapters (list of dict): Each chapter has 'heading' and 'content' or 'list'.
    """
    pdf = ResearchReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Add main title
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, title, ln=1, align='C')
    pdf.ln(10)

    # Add chapters
    for chapter in chapters:
        pdf.chapter_title(chapter["heading"])
        if "content" in chapter:
            pdf.chapter_body(chapter["content"])
        if "list" in chapter:
            pdf.add_list(chapter["list"])

    # Save the PDF
    pdf.output(filename)

# Example usage
chapters = [
    {
        "heading": "Introduction",
        "content": "A research report is used to convey information about a study..."
    },
    {
        "heading": "Steps to Writing a Research Report",
        "list": [
            "Analyse the task.",
            "Develop a rough plan.",
            "Do the research.",
            "Draft the body of the report.",
            "Draft the supplementary material.",
            "Draft the preliminary material.",
            "Polish your report."
        ]
    },
    {
        "heading": "Key Sections of a Research Report",
        "content": "A typical research report includes preliminary material, body, and supplementary material..."
    }
]

generate_research_guide_pdf("Research_Report_Guide.pdf", "Writing a Research Report", chapters)
