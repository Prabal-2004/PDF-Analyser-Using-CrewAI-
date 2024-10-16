import os
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import streamlit as st
import PyPDF2
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
import logging
import base64
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMAIL_HOST = "smtp.gmail.com"
EMAIL_PORT = 587
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
RECIPIENT_EMAIL = "piyush.bhawsar1@gmail.com"
COMPANY_NAME = "Wingify"
LOGO_PATH = "logo.png"

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Existing helper functions (extract_text_from_pdf, search_and_crawl, send_professional_email)
# ... (keep these functions as they are in the original code)

# Define Agents
class PDFAnalyzer(Agent):
    def analyze_pdf(self, pdf_text):
        analysis_prompt = """
        Analyze the following PDF content and provide a detailed summary covering:
        1. Main topic or purpose
        2. Key points or arguments
        3. Significant data or statistics
        4. Intended audience
        5. Overall tone and style
        6. Any notable findings or insights
        Provide a comprehensive analysis in a well-structured format.
        """
        return self.execute_task(analysis_prompt, pdf_text)

class WebResearcher(Agent):
    def search_and_analyze(self, query):
        crawled_data, crawled_urls = search_and_crawl(query)
        return crawled_data, crawled_urls

class FinalAnalyzer(Agent):
    def synthesize_analysis(self, pdf_analysis, web_data):
        final_analysis_prompt = """
        Based on the PDF analysis and the crawled web data, provide a comprehensive summary that:
        1. Synthesizes the main ideas from both the PDF and web sources
        2. Highlights any new insights or perspectives gained from the additional sources
        3. Identifies any contradictions or confirmations between the PDF content and web data
        4. Suggests potential areas for further research or exploration
        5. Provides actionable recommendations based on the overall analysis
        Ensure the analysis is thorough, well-structured, and provides valuable insights.
        """
        return self.execute_task(final_analysis_prompt, f"PDF Analysis:\n{pdf_analysis}\n\nCrawled Web Data:\n{web_data}")

# Define Tasks
def analyze_pdf_task(agent, pdf_text):
    return Task(
        description="Analyze the content of the PDF",
        agent=agent,
        context={"pdf_text": pdf_text}
    )

def web_research_task(agent, query):
    return Task(
        description="Perform web research based on PDF content",
        agent=agent,
        context={"query": query}
    )

def final_analysis_task(agent, pdf_analysis, web_data):
    return Task(
        description="Synthesize PDF analysis and web research",
        agent=agent,
        context={"pdf_analysis": pdf_analysis, "web_data": web_data}
    )

# Streamlit UI
def main():
    st.set_page_config(page_title="Wingify PDF Analyzer", layout="wide")
    st.title("📊 Wingify PDF Analyzer and Reporting System")

    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

    if uploaded_file is not None:
        st.success("File successfully uploaded!")
        
        with st.expander("Preview PDF"):
            display_pdf(uploaded_file)
        
        if st.button("Analyze PDF"):
            try:
                with st.spinner("Analyzing your PDF... This may take a few minutes."):
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Step 1: Extract text from PDF
                    status_text.text("Extracting text from PDF...")
                    pdf_text = extract_text_from_pdf(uploaded_file)
                    progress_bar.progress(20)

                    # Initialize Crew and Agents
                    pdf_analyzer = PDFAnalyzer(name="PDF Analyzer", openai_api_key=OPENAI_API_KEY)
                    web_researcher = WebResearcher(name="Web Researcher", openai_api_key=OPENAI_API_KEY)
                    final_analyzer = FinalAnalyzer(name="Final Analyzer", openai_api_key=OPENAI_API_KEY)

                    crew = Crew(
                        agents=[pdf_analyzer, web_researcher, final_analyzer],
                        tasks=[
                            analyze_pdf_task(pdf_analyzer, pdf_text),
                            web_research_task(web_researcher, ""),
                            final_analysis_task(final_analyzer, "", "")
                        ],
                        verbose=2
                    )

                    # Execute tasks
                    results = crew.kickoff()

                    # Extract results
                    content_analysis = results[0]
                    crawled_data, crawled_urls = results[1]
                    final_analysis = results[2]

                    progress_bar.progress(80)

                    # Send email report
                    status_text.text("Preparing and sending email report...")
                    original_filename = uploaded_file.name
                    analysis_filename = f"wingify_analysis_{original_filename}"
                    with open(analysis_filename, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    send_professional_email(
                        f"Wingify PDF Analysis Report: {original_filename}",
                        content_analysis,
                        crawled_urls,
                        final_analysis,
                        analysis_filename
                    )
                    os.remove(analysis_filename)
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")

                # Display results
                st.success("✅ Analysis completed successfully!")
                st.write("### PDF Content Analysis")
                st.write(content_analysis)
                
                st.write("### Crawled Websites")
                for url in crawled_urls:
                    st.write(f"- {url}")
                
                st.write("### Comprehensive Final Analysis")
                st.write(final_analysis)

                st.success(f"A detailed report has been sent to {RECIPIENT_EMAIL}")

            except Exception as e:
                st.error(f"An error occurred during the analysis: {str(e)}")
                logging.error(f"Analysis failed: {str(e)}")

    st.sidebar.title("About Wingify PDF Analyzer")
    st.sidebar.info(
        "This advanced PDF Analyzer uses AI to extract insights from your documents, "
        "cross-reference with the latest web data, and provide comprehensive analysis. "
        "Upload a PDF to get started!"
    )

    # Display logs in the sidebar
    with st.sidebar.expander("View Logs"):
        if st.button("Refresh Logs"):
            log_output = st.empty()
            with open("app.log", "r") as log_file:
                log_output.text(log_file.read())

if __name__ == "__main__":
    main()