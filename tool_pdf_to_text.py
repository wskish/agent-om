from pydantic import Field, HttpUrl
import asyncio
import aiohttp
from io import BytesIO
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from ai_toolchat import BaseToolParam, ToolMessage
from loguru import logger

# Define a pydantic model based on BaseToolParam to define the parameters for the tool
class PDFToTextParam(BaseToolParam):
    url: str = Field(description="The URL of the PDF to convert to text.")

# Function to extract text from PDF bytes using pdfminer
def pdf_to_text_via_pdfminer(pdf_bytes: bytes) -> str:
    """
    Extract text from the pdf_bytes and return it as a single string.
    """
    text = ""
    pagecount = 0
    
    for page_layout in extract_pages(BytesIO(pdf_bytes)):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    line = text_line.get_text().rstrip() + "\n"
                    text += line    
    # Remove null characters which sometimes occur but the db doesn't like
    text = text.replace('\x00', '')                    
    return text

# Define the tool function
async def pdf_to_text(param: PDFToTextParam):
    """
    This tool downloads a PDF from the provided URL and converts it to text using pdfminer.
    Use this tool when the user provides a PDF URL and requests to extract its text content.
    """
    yield ToolMessage(f"Downloading PDF from {param.url}...")

    # Download the PDF file
    async with aiohttp.ClientSession() as session:
        async with session.get(param.url) as response:
            if response.status != 200:
                err = f"Failed to download PDF from {param.url}. HTTP status: {response.status}"
                logger.warning(err)
                raise ValueError(err)
            pdf_bytes = await response.read()

    yield ToolMessage(f"Converting PDF to text...")

    # Convert the PDF bytes to text
    try:
        text = pdf_to_text_via_pdfminer(pdf_bytes)
    except Exception as e:
        err = f"Error converting PDF to text: {str(e)}"
        logger.error(err)
        raise ValueError(err)

    # Return the extracted text
    yield text
