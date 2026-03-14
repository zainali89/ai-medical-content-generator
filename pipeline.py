"""
LangGraph pipeline: node functions and workflow graph for the article-generation pipeline.
"""

import datetime
import json
import logging
import os
import tempfile
import urllib.parse

import PyPDF2
import requests
from docx import Document
from google.genai import types
from langgraph.graph import END, StateGraph

from config import (
    PERPLEXITY_API_KEY,
    SCHEDULER_API_URL,
    firecrawl_client,
    genai_client,
    openai_client,
)
from models import State
from utils import get_youtube_transcript, timeit

logger = logging.getLogger("ai_medical")


# ============================================================================
# Node functions
# ============================================================================

@timeit
def process_user_input(state: State) -> dict:
    logger.info(f"Processing user input: {state['user_input_topic']}")
    topic = state["user_input_topic"]
    corrected_topic = topic.title()
    skip_perplexity = bool(
        state["reference_urls"] or state["docs_files"] or state["youtube_links"]
    )
    logger.info(f"Using topic: '{corrected_topic}', skip_perplexity: {skip_perplexity}")
    return {
        "user_input_topic": corrected_topic,
        "skip_perplexity": skip_perplexity,
        "errors": [],
        "performance_metrics": {},
        "critical_error": False,
    }


@timeit
def search_perplexity(state: State) -> dict:
    logger.info(f"Searching Perplexity for: {state['user_input_topic']}")
    perplexity_url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "sonar-reasoning",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a specialized medical research assistant with expertise in searching "
                    "and retrieving highly technical, clinical and research-focused information from "
                    "medical literature. You MUST retrieve information ONLY from: "
                    "1. Peer-reviewed medical journals with high impact factors (e.g., NEJM, The Lancet, JAMA, BMJ) "
                    "2. Official clinical guidelines from recognized health organizations (WHO, CDC, NIH, etc.) "
                    "3. Medical academic institutions and teaching hospitals "
                    "4. Specialized medical databases (PubMed, Cochrane Library, etc.) "
                    "IMPORTANT REQUIREMENTS: "
                    "- Focus exclusively on scientifically validated, evidence-based medical information "
                    "- Include specific medical terminology, diagnostic criteria, treatment protocols, and clinical outcomes "
                    "- Cite recent research (within last 3-5 years when available) "
                    "- Provide detailed statistics, study methodologies, and findings "
                    "- Include information on current consensus and areas of ongoing research "
                    "- NEVER fabricate or extrapolate beyond what is explicitly stated in reliable sources "
                    "- NEVER reuse information from previous queries "
                    "- RESET your memory and knowledge for each new query to prevent contamination "
                    "Your responses must meet professional medical standards and be suitable for clinical or academic use."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conduct a thorough search for specialized medical information on "
                    f"{state['user_input_topic']} focusing specifically on: "
                    f"'{state['user_input_description']}'\n\n"
                    f"Target information for audience: {state['target_audience']}\n\n"
                    "REQUIREMENTS:\n"
                    "1. Begin with a concise summary of current clinical understanding and research status\n"
                    "2. Provide detailed medical information including:\n"
                    "   - Precise diagnostic criteria and classifications\n"
                    "   - Evidence-based treatment approaches with efficacy data\n"
                    "   - Pathophysiology and mechanisms of action\n"
                    "   - Epidemiological data and relevant statistics\n"
                    "   - Current clinical guidelines and standard of care\n"
                    "   - Recent advances, trials, or novel approaches\n"
                    "   - Areas of medical consensus vs. controversy\n"
                    "3. Structure your response in clearly labeled clinical subsections\n"
                    "4. Include ONLY facts that can be verified through medical literature\n"
                    "5. End with a comprehensive reference list in this format ONLY:\n"
                    "   [Number] Title (Author(s), Publication Date). Link: [direct URL to medical source]\n\n"
                    f"Focus EXCLUSIVELY on {state['user_input_topic']}."
                ),
            },
        ],
        "max_tokens": 3500,
    }
    errors = []
    critical_error = False
    try:
        response = requests.post(perplexity_url, headers=headers, json=data)
        response.raise_for_status()
        perplexity_data = response.json()["choices"][0]["message"]["content"].split("\n")
        logger.info(f"Retrieved {len(perplexity_data)} lines from Perplexity")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code in [401, 403]:
            errors.append("Critical error: Invalid Perplexity API key")
            critical_error = True
            logger.error("Critical error: Invalid Perplexity API key")
        else:
            errors.append(f"Perplexity error: {e}")
            logger.error(f"Perplexity search failed: {e}")
        perplexity_data = []
    except requests.RequestException as e:
        errors.append(f"Perplexity error: {e}")
        perplexity_data = []
        logger.error(f"Perplexity search failed: {e}")
    return {
        "perplexity_data": perplexity_data,
        "errors": errors,
        "performance_metrics": {},
        "critical_error": critical_error,
    }


@timeit
def extract_firecrawl_content(state: State) -> dict:
    if not state["reference_urls"]:
        logger.info("No reference URLs provided, skipping Firecrawl extraction")
        return {
            "firecrawl_data": [],
            "errors": [],
            "performance_metrics": {},
            "critical_error": False,
        }

    logger.info(f"Extracting Firecrawl content for {len(state['reference_urls'])} URLs")
    errors = []
    firecrawl_data = []

    for url in state["reference_urls"]:
        try:
            scraped = firecrawl_client.scrape_url(url)
            if isinstance(scraped, dict):
                content = scraped.get("markdown", "No content found")
            else:
                content = getattr(scraped, "markdown", "No content found")
                if content == "No content found" and hasattr(scraped, "content"):
                    content = scraped.content or "No content found"
            firecrawl_data.append(f"Firecrawl content from {url}: {content}")
            logger.info(f"Successfully extracted Firecrawl content from {url}")
        except Exception as e:
            error_msg = f"Firecrawl extraction error for {url}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)

    return {
        "firecrawl_data": firecrawl_data,
        "errors": errors,
        "performance_metrics": {},
        "critical_error": False,
    }


@timeit
def process_docs(state: State) -> dict:
    if not state["docs_files"]:
        logger.info("No document files provided, skipping docs processing")
        return {
            "docs_data": [],
            "errors": [],
            "performance_metrics": {},
            "critical_error": False,
        }

    logger.info(f"Processing {len(state['docs_files'])} document files")
    errors = []
    docs_data = []

    for doc_path in state["docs_files"]:
        file_path = doc_path
        is_url = doc_path.startswith("http://") or doc_path.startswith("https://")
        try:
            if is_url:
                response = requests.get(doc_path, stream=True)
                response.raise_for_status()

                content_disposition = response.headers.get("content-disposition")
                if content_disposition and "filename=" in content_disposition:
                    filename = content_disposition.split("filename=")[1].strip('"')
                else:
                    filename = urllib.parse.urlparse(doc_path).path.split("/")[-1]

                if not filename.lower().endswith((".pdf", ".docx")):
                    raise ValueError(f"Unsupported file format in URL: {filename}")

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(filename)[1]
                ) as temp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            temp_file.write(chunk)
                    file_path = temp_file.name
                logger.info(f"Downloaded document from {doc_path} to temporary file: {file_path}")

            file_ext = os.path.splitext(file_path)[1].lower()
            text = ""

            if file_ext == ".pdf":
                with open(file_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + " "
                text = text.strip()
                if not text:
                    raise ValueError("No text extracted from PDF")
                logger.info(f"Successfully extracted text from PDF: {doc_path}")

            elif file_ext == ".docx":
                document = Document(file_path)
                for para in document.paragraphs:
                    if para.text.strip():
                        text += para.text + " "
                text = text.strip()
                if not text:
                    raise ValueError("No text extracted from DOCX")
                logger.info(f"Successfully extracted text from DOCX: {doc_path}")

            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

            docs_data.append(f"Document: {text}")

            if is_url:
                os.unlink(file_path)
        except Exception as e:
            errors.append(f"Document processing error for {doc_path}: {e}")
            logger.error(f"Failed to process document {doc_path}: {e}")
            if is_url and os.path.exists(file_path):
                os.unlink(file_path)

    return {
        "docs_data": docs_data,
        "errors": errors,
        "performance_metrics": {},
        "critical_error": False,
    }


@timeit
def process_youtube_links(state: State) -> dict:
    if not state["youtube_links"]:
        logger.info("No YouTube links provided, skipping YouTube processing")
        return {
            "youtube_data": [],
            "errors": [],
            "performance_metrics": {},
            "critical_error": False,
        }

    logger.info(f"Processing {len(state['youtube_links'])} YouTube links")
    results = get_youtube_transcript(state["youtube_links"])
    youtube_data = []
    errors = []
    for result in results:
        if result.get("error"):
            errors.append(f"YouTube transcript error for {result['video_url']}: {result['error']}")
        else:
            youtube_data.append(f"Transcript from {result['video_url']}: {result['transcript_text']}")

    return {
        "youtube_data": youtube_data,
        "errors": errors,
        "performance_metrics": {},
        "critical_error": False,
    }


@timeit
def generate_content(state: State) -> dict:
    if state["critical_error"]:
        logger.error("Critical error detected, skipping content generation")
        return {
            "generated_content": "",
            "errors": ["Critical error occurred, content generation skipped"],
            "performance_metrics": {},
            "critical_error": True,
        }

    logger.info(f"Generating content for: {state['user_input_topic']}")

    if (
        not state["perplexity_data"]
        and not state["firecrawl_data"]
        and not state["docs_data"]
        and not state["youtube_data"]
    ):
        logger.warning("No data available for content generation")
        return {
            "generated_content": "",
            "errors": ["No data available"],
            "performance_metrics": {},
            "critical_error": True,
        }

    current_date = datetime.date.today()

    perplexity_data = (
        None
        if state["skip_perplexity"]
        else "\n".join([f"Perplexity: {item}" for item in state["perplexity_data"] if item])
    )
    firecrawl_data = "\n".join([f"Firecrawl: {item}" for item in state["firecrawl_data"] if item])
    docs_data = "\n".join([f"Document: {item}" for item in state["docs_data"] if item])
    youtube_data = "\n".join([f"YouTube: {item}" for item in state["youtube_data"] if item])

    length_mapping = {"Short": 800, "Medium": 1200, "Long": 2200}
    length_words = length_mapping.get(state["article_length"], 800)

    prompt = f"""
    Write a referenced, fact-checked, and neutral article about {state['user_input_topic']} specifically tailored for {state['target_audience']}. Use Australian English (e.g., 'organise', 'centre') and base all factual claims STRICTLY on the provided reference data from peer-reviewed or credible sources.

    ARTICLE FORMAT:
    - Title: Use ONLY "{state['user_input_topic']}" as the title. Do not modify, expand, or rewrite this title.
    - Do NOT repeat the topic as an introduction paragraph or summary at the beginning of the article.
    - Start directly with relevant content after the title.
    - Be concise and prioritize completion over verbose explanations.

    CRITICAL REQUIREMENT: ALL ARTICLES MUST INCLUDE A COMPLETE REFERENCES SECTION AT THE END. This is non-negotiable.
    Your response will be rejected if references are missing or incomplete. Reserve at least 10% of your word count for references.

    IMPORTANT: The article MUST be EXACTLY {length_words} words in length (+/-10%) INCLUDING the references section.
    Structure your article to fit this length requirement, ensuring references are never cut off.

    TO PREVENT CUTOFFS: Be more concise in your explanations, use fewer examples, and ensure you have enough space for the references section.
    DO NOT leave any sentences unfinished.

    IMPORTANT: DO NOT HALLUCINATE OR INVENT ANY INFORMATION. If the provided reference data doesn't cover a particular aspect of the topic, explicitly state that information is limited rather than making up facts. Only include information that is directly supported by the reference data provided below.

    Adjust language and detail for the audience:
    - Medical Professionals (Doctors): Employ precise medical terminology and provide comprehensive, detailed analysis.
    - Students: Utilize technical medical vocabulary and deliver thorough, educational analysis.
    - General Public: Use simple, everyday words, clarify any complex terms, and highlight useful, easy-to-apply information.
    - Patients: Use clear, straightforward language, explain medical terms simply, and emphasise practical, health-related advice

    Use the reference data below to support claims, ensuring the article is engaging and accessible. The data includes:
    - **Perplexity**: Text with a 'Sources' section (e.g., 'Title (Author(s), Date). Link: [URL]'). Use URLs exactly as provided.
    - **Firecrawl**: Scraped content prefixed with source URL (e.g., 'Website content from [URL]: [content]'). Use the URL provided in the prefix.
    - **Documents**: Content extracted from document files, prefixed with 'Document:'.
    - **YouTube**: Content from YouTube videos, prefixed with 'YouTube:'.

    Rules for references and content:
    - Extract Perplexity URLs from lines like 'Link: [URL]' and use them unchanged.
    - Extract Website URLs from the prefix 'Website content from [URL]' and use them unchanged.
    - For Document content, cite as "From document analysis" if no specific citation is available.
    - For YouTube content, cite as "From [YouTube video title]" if available.
    - Include a reference only if it has a valid URL from the data. If no URL exists, omit it -- do NOT invent links (e.g., no '.example.com').
    - Verify all facts against the data and correct errors.
    - Always make sure the links are clickable.
    - Only include the links in the references.
    - If you're uncertain about any information, indicate this clearly rather than guessing.
    - For any statistical claims, medical recommendations, or specific treatments, cite the exact source from the reference data.

    Keep the tone objective and evidence-based, current as of {current_date}, and note missing data if applicable.

    STRUCTURE OF YOUR RESPONSE:
    1. Title: ONLY "{state['user_input_topic']}" (not prefixed with "Medical Topic:" or any other text)
    2. Main article content (start immediately with relevant information, be concise)
    3. Mandatory "References" heading
    4. Complete numbered reference list in this format ONLY:
       - [Number]. Title (Author(s), Date). Link: [URL]

    EVERY reference you cite in-text MUST appear in the references section. Reserve AT LEAST 10% of your word count for references.
    Double-check that your response ends with complete references before submitting.

    - User Description: {state['user_input_description']}
    - Length: ~{length_words} words (INCLUDING references)
    - Reference Data (Perplexity):
    {perplexity_data if perplexity_data else 'No Perplexity data available'}
    - Reference Data (Website):
    {firecrawl_data if firecrawl_data else 'No Firecrawl data available'}
    - Reference Data (Documents):
    {docs_data if docs_data else 'No Document data available'}
    - Reference Data (YouTube):
    {youtube_data if youtube_data else 'No YouTube data available'}

    If Reference Data (Website), (Documents), or (YouTube) has any kind of data, then Perplexity data should be ignored and set to None.
    """

    errors = []
    critical_error = False
    try:
        estimated_tokens = int(length_words * 2.0)
        reserved_tokens = 1000
        max_tokens = min(8000, max(4000, estimated_tokens * 2 + reserved_tokens))

        response_stream = genai_client.models.generate_content_stream(
            model="gemini-2.5-flash-preview-04-17",
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(
                temperature=0.3,
                top_p=0.9,
                max_output_tokens=max_tokens,
            ),
        )

        response_chunks = []
        for chunk in response_stream:
            response_chunks.append(chunk.text)
        content = "".join(response_chunks).strip()

        word_count = len(content.split())
        target_words = length_words
        min_words = int(target_words * 0.9)
        max_words = int(target_words * 1.1)

        if min_words <= word_count <= max_words:
            logger.info(
                f"Content generated successfully using Gemini - Word count: {word_count} (target: {target_words})"
            )
        else:
            logger.warning(
                f"Generated content doesn't meet length requirements - "
                f"Got {word_count} words, target was {target_words} (+/-10%)"
            )

        logger.info("Content generated successfully using Gemini")
    except Exception as e:
        errors.append(f"Gemini API error: {e}")
        content = ""
        critical_error = True
        logger.error(f"Content generation failed with Gemini: {e}")
    return {
        "generated_content": content,
        "errors": errors,
        "performance_metrics": {},
        "critical_error": critical_error,
    }


@timeit
def validate_content(state: State) -> dict:
    if state["critical_error"]:
        logger.error("Critical error detected, skipping content validation")
        return {
            "errors": ["Critical error occurred, validation skipped"],
            "performance_metrics": {},
            "critical_error": True,
        }
    logger.info("Validating content")
    if not state["generated_content"]:
        logger.warning("No content to validate")
        return {
            "errors": ["No content available"],
            "performance_metrics": {},
            "critical_error": True,
        }
    current_date = datetime.date.today()
    validation_prompt = f"""
    You are a medical content validator tasked with reviewing a medical article for general quality.
    Your goal is to determine if the article is suitable for use, with a focus on being reasonably lenient while ensuring basic standards are met.
    Below is the article to validate:

    Article Content:
    {state['generated_content']}

    Validate the article based on the following criteria, but apply these criteria with flexibility:

    1. **General Accuracy**: Check if the content is broadly accurate and consistent with common medical knowledge on the topic "{state['user_input_topic']}".
    Allow for minor inaccuracies or generalizations as long as they do not fundamentally misrepresent the topic or pose a risk of harm.

    2. **AMA Citation Format**: Verify that citations are present and generally follow the AMA format.
    Be lenient with minor formatting issues as long as the citations are recognizable.

    3. **Appropriateness for Audience**: Ensure the content is reasonably suitable for the target_audience, which is "{state['target_audience']}".

    Validation Guidelines:
    - Mark the article as "Valid" if it meets the above criteria in a general sense, even if there are minor issues.
    - Mark the article as "Invalid" only if there are significant issues.
    - Provide a list of specific issues (if any).

    Return a JSON object with the following structure:
    {{
        "status": "Valid" or "Invalid",
        "issues": ["list of specific issues or empty list if Valid"]
    }}
    """
    errors = []
    critical_error = False
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": validation_prompt}],
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        validation_data = json.loads(response.choices[0].message.content)
        if validation_data.get("status") != "Valid":
            errors.extend(validation_data.get("issues", []))
            logger.warning(f"Validation failed: {validation_data.get('issues', [])}")
        else:
            logger.info("Content validated successfully")
    except Exception as e:
        errors.append(f"Validation error: {e}")
        critical_error = True
        logger.error(f"Validation failed: {e}")
    return {
        "errors": errors,
        "performance_metrics": {},
        "critical_error": critical_error,
    }


def check_data_availability(state: State) -> dict:
    logger.info("Checking data availability")
    if state["critical_error"]:
        logger.warning("Critical error detected, but proceeding with available data")
    return {
        "errors": [],
        "performance_metrics": {},
        "critical_error": state["critical_error"],
    }


# ============================================================================
# Routing helpers
# ============================================================================

def route_after_process_user_input(state: State) -> str:
    if state["skip_perplexity"]:
        return "extract_firecrawl_content"
    return "search_perplexity"


def route_after_check_data(state: State) -> str:
    return "generate_content"


# ============================================================================
# Build the LangGraph workflow
# ============================================================================

def build_workflow():
    """Construct and compile the LangGraph state machine."""
    workflow = StateGraph(State)
    workflow.add_node("process_user_input", process_user_input)
    workflow.add_node("search_perplexity", search_perplexity)
    workflow.add_node("extract_firecrawl_content", extract_firecrawl_content)
    workflow.add_node("process_docs", process_docs)
    workflow.add_node("process_youtube_links", process_youtube_links)
    workflow.add_node("check_data_availability", check_data_availability)
    workflow.add_node("generate_content", generate_content)
    workflow.add_node("validate_content", validate_content)

    workflow.set_entry_point("process_user_input")
    workflow.add_conditional_edges(
        "process_user_input",
        route_after_process_user_input,
        {
            "search_perplexity": "search_perplexity",
            "extract_firecrawl_content": "extract_firecrawl_content",
        },
    )
    workflow.add_edge("search_perplexity", "extract_firecrawl_content")
    workflow.add_edge("extract_firecrawl_content", "process_docs")
    workflow.add_edge("process_docs", "process_youtube_links")
    workflow.add_edge("process_youtube_links", "check_data_availability")
    workflow.add_conditional_edges(
        "check_data_availability",
        route_after_check_data,
        {"generate_content": "generate_content"},
    )
    workflow.add_edge("generate_content", "validate_content")
    workflow.add_edge("validate_content", END)

    return workflow.compile()


langgraph_app = build_workflow()
