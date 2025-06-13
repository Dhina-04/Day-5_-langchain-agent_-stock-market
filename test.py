import os
import streamlit as st
from langchain.tools import DuckDuckGoSearchResults
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

# 🔐 Set your Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCEqjX9VInQKeSlFCOXeP653_2FNWL5CW4"

# 🧠 Gemini 2.0 Flash LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# 🔍 Search Tool
search_tool = DuckDuckGoSearchResults()

# 📜 Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial assistant that extracts the current Indian stock price from the search results."),
    ("user", "Here are the search results for '{stock_name}':\n\n{search_results}\n\nBased on this, what is the current stock price?")
])

# 🔗 Create the chain
chain: Runnable = (
    {
        "stock_name": lambda x: x["stock_name"],
        "search_results": lambda x: search_tool.run(x["stock_name"] + " current stock price India")
    }
    | prompt
    | llm
)

# 🎨 Streamlit UI
st.set_page_config(page_title="📈 Indian Stock Price Checker")
st.title("📈 Indian Stock Price Checker")
st.markdown("Enter the name of an Indian stock (e.g., *Reliance*, *Infosys*, *TCS*) to get the latest price using Gemini 2.0 Flash.")

stock_name = st.text_input("Enter Stock Name")

if st.button("Get Stock Price"):
    if not stock_name.strip():
        st.warning("Please enter a valid stock name.")
    else:
        try:
            with st.spinner("Fetching latest stock price..."):
                result = chain.invoke({"stock_name": stock_name})
                st.success("Stock price retrieved successfully!")
                st.markdown(f"**Result:**\n\n{result.content}")
        except Exception as e:
            st.error(f"Error: {e}")
