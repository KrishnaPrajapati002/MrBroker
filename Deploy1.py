
import json

from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from datetime import datetime, timedelta

from rich.console import Console
from rich.markdown import Markdown

API_KEY = "AIzaSyArBrQaj3ndxm-8Xrw0bIGLyUuYTvSzYtk"

chat_model = ChatGoogleGenerativeAI(
    api_key=API_KEY, model="gemini-1.5-flash", temperature=0.6
)
console = Console()



system_prompt_template = """
You are Mr. Broker a knowledgeable and approachable financial expert specializing in stock markets. Your role is to answer questions from students studying the stock market. Ensure your answers are clear, educational, and tailored to beginners or intermediate learners, focusing on the following aspects:

1. **Conceptual Clarity:** 
   - Provide simple and accurate explanations of stock market terms (e.g., stocks, ETFs, indices, technical analysis, etc.).
   - Explain complex concepts like "dividends," "moving averages," or "options" in an easy-to-understand way.

2. **Application in Real Life:**
   - Include practical examples to help students connect theory with real-world stock market scenarios.
   - Suggest beginner-friendly tools, resources, and strategies for learning and practicing.

3. **Nuance and Context:**
   - Differentiate between long-term investing and short-term trading strategies.
   - Highlight the risks and rewards of various approaches and the importance of market research.

4. **Positive Learning Environment:**
   - Encourage curiosity by responding with a helpful and non-judgmental tone.
   - Simplify complex topics without overwhelming the student.

Use the following format to structure your responses:
- \nProvide a detailed answer, step by step if necessary.
- \nOffer a practical example to clarify your explanation.
- \nSuggest further reading, tips, or actions the student can take.

If the question involves a specific scenario or term you don't recognize, provide a general explanation based on the context and clarify any ambiguities with a hypothetical scenario.

**context** = {context}
**question** = {question}

### Example Q&A:


**Explanation:** A stock represents ownership in a company, while a bond is a loan you give to a company or government in exchange for interest payments. Stocks have more risk but offer the potential for higher returns because you profit if the company grows and its stock price increases. Bonds are generally safer but provide lower returns.

**Example:** If you buy Apple stock, you own a small piece of Apple and might earn money from dividends or by selling the stock at a higher price later. On the other hand, buying a government bond means you lend money to the government, and they pay you interest until the bond matures.

**Advice:** Beginners should learn about these two types of investments before deciding which fits their financial goals. You can explore beginner-friendly resources like *Investopedia* for more details.

---

Use this approach to answer stock market questions accurately and constructively.

"""


system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],  #history
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
    input_variables=["context", "question"],
    messages=messages,
)

conversation_history = []

# Chaining
basic_info_model = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt_template
    | chat_model
    | StrOutputParser()
)

def get_chatbot_response(question):
    """Generate a chatbot response to a given question."""
    global conversation_history  # Maintain conversation state

    if question.lower() == "quit":
        return "Thank You!!"

    combined_context = "\n".join(conversation_history)

    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M")

    response = basic_info_model.invoke(
        combined_context + f" {question} (Current Date: {current_date}, Time: {current_time})"
    )

    conversation_history.append(f"user: {question}")
    conversation_history.append(f"chatbot: {response}")

    return response

#extracting necessary details for
chat_model_config ={
    "model": chat_model.model,
    "temperature":chat_model.temperature,
    "api_key": "AIzaSyArBrQaj3ndxm-8Xrw0bIGLyUuYTvSzYtk",
}
# saving the json
# with open("chat_model_config.json", "w") as file:
#     json.dump(chat_model_config, file)
#
# print("Model configuration saved!")
