import os
import streamlit as st
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

# 🔐 Set Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCEqjX9VInQKeSlFCOXeP653_2FNWL5CW4"

# 🌟 Gemini 2.0 Flash Model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# 🔍 DuckDuckGo search tool
search_tool = DuckDuckGoSearchResults()

# 📜 Prompt Template (Improved)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial assistant. Use the search results to find the most recent available stock price for the company."),
    ("user", "Here are the search results for '{stock_name}':\n\n{search_results}\n\nWhat is the most recently available stock price?")
])

# 🔗 LangChain Runnable chain
chain: Runnable = (
    {
        "stock_name": lambda x: x["stock_name"],
        "search_results": lambda x: search_tool.run(x["stock_name"] + " share price India")
    }
    | prompt
    | llm
)

# 🎨 Streamlit UI
st.set_page_config(page_title="📊 Indian Stock Price Checker")
st.title("📊 Check Latest Indian Stock Price (Approx.)")
st.markdown("Enter the name of an Indian stock (e.g., **TCS**, **Infosys**, **Reliance**) to get the most recently available share price from public sources.")

stock_name = st.text_input("Enter Stock Name")

if st.button("Get Stock Price"):
    if not stock_name.strip():
        st.warning("Please enter a valid stock name.")
    else:
        try:
            with st.spinner("Searching..."):
                result = chain.invoke({"stock_name": stock_name})
                st.success("Here’s the most recent available stock info:")
                st.markdown(f"**Response:**\n\n{result.content}")
                st.markdown("*🔔 Note: This is not guaranteed to be real-time. For official data, please check NSE/BSE.*")
        except Exception as e:
            st.error(f"❌ Error: {e}")
