import os
import re
import json
import yfinance as yf
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv

# Load API Key from environment variable
load_dotenv()
# API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
# if not API_KEY:
#     raise ValueError("API Key for Google Generative AI is missing! Set it as an environment variable.")

# Initialize chat model
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.6)
console = Console()

# System prompt template
system_prompt_template = """
You are StockMentor, a knowledgeable and approachable financial expert specializing in stock markets...
---
Stock Analysis:
- Ticker: {ticker}
- Latest Closing Price: {closing_price}
- 200-Day Moving Average: {dma_value}
---
"""

system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context", "closing_price", "dma_value", "ticker"],
        template=system_prompt_template,
    )
)

human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)

messages = [system_prompt, human_prompt]

prompt_template = ChatPromptTemplate(
    input_variables=["context", "question", "closing_price", "dma_value", "ticker"],
    messages=messages,
)

conversation_history = []

basic_info_model = (
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough(),
            "closing_price": RunnablePassthrough(),
            "dma_value": RunnablePassthrough(),
            "ticker": RunnablePassthrough(),
        }
        | prompt_template
        | chat_model
        | StrOutputParser()
)


def extract_ticker(question):
    match = re.search(r'\b[A-Z]{1,5}(\.[A-Z]{2,3})?\b', question)
    return match.group(0) if match else None


def perform_stock_analysis(ticker):
    try:
        stock_data = yf.Ticker(ticker).history(period="1y")
        if stock_data.empty:
            return None, None, f"No data retrieved for ticker '{ticker}'."

        stock_data["200DMA"] = stock_data["Close"].rolling(window=200).mean()
        dma_value = stock_data["200DMA"].iloc[-1] if len(stock_data) >= 200 else None
        closing_price = stock_data["Close"].iloc[-1]

        if dma_value is None or pd.isna(dma_value):
            return closing_price, None, f"Not enough data to calculate the 200 DMA for '{ticker}'."

        suggestion = "Buy" if closing_price > dma_value * 1.03 else "Sell" if closing_price < dma_value * 0.97 else "Hold"
        return closing_price, dma_value, suggestion
    except Exception as e:
        return None, None, f"Error during stock analysis: {e}"


def get_chatbot_response(question):
    ticker = extract_ticker(question)
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M")
    combined_context = "\n".join(conversation_history)

    if ticker:
        closing_price, dma_value, suggestion = perform_stock_analysis(ticker)
        if suggestion is None:
            response = suggestion  # Error message
        else:
            response = f"Ticker: {ticker}\nLatest Closing Price: {closing_price}\n200-DMA: {dma_value}\nSuggestion: {suggestion}"
    else:
        final_prompt = prompt_template.format(
            context=combined_context,
            question=question,
            ticker="N/A",
            closing_price="N/A",
            dma_value="N/A",
        )
        response = basic_info_model.invoke(final_prompt)

    conversation_history.append(f"user: {question}")
    conversation_history.append(f"chatbot: {response}")
    return response
