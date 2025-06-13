import os
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

# Set Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCEqjX9VInQKeSlFCOXeP653_2FNWL5CW4"

# Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# DuckDuckGo Tool
search_tool = DuckDuckGoSearchResults()

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial assistant. Use the search results to find the most recent available stock price for the company."),
    ("user", "Here are the search results for '{stock_name}':\n\n{search_results}\n\nWhat is the most recently available stock price?")
])

# LangChain Chain
chain: Runnable = (
    {
        "stock_name": lambda x: x["stock_name"],
        "search_results": lambda x: search_tool.run(x["stock_name"] + " share price India")
    }
    | prompt
    | llm
)

# Streamlit UI
st.set_page_config(page_title="📊 Indian Stock Dashboard")
st.title("📊 Indian Stock Info + 52-Week Chart")
st.markdown("Enter a stock name (e.g., **TCS**, **Infosys**) to get the latest price and chart of 52-week highs and lows.")

stock_name = st.text_input("Enter Stock Name")
stock_symbol = st.text_input("Enter NSE Symbol (e.g., INFY.NS, RELIANCE.NS)", help="Use Yahoo Finance NSE code (e.g., TCS.NS, HDFCBANK.NS)")

if st.button("Get Stock Info"):
    if not stock_name or not stock_symbol:
        st.warning("Please enter both stock name and NSE symbol.")
    else:
        try:
            # LangChain + Gemini Summary
            with st.spinner("Getting stock summary..."):
                result = chain.invoke({"stock_name": stock_name})
                st.success("Stock summary found!")
                st.markdown(f"**Summary:**\n\n{result.content}")
                st.markdown("*Note: Not real-time. For official info, check NSE/BSE.*")

            # 52-Week Chart
            with st.spinner("Fetching 1-year historical data..."):
                data = yf.download(stock_symbol, period="1y")
                if data.empty:
                    st.warning("Could not fetch data. Check symbol.")
                else:
                    high_52 = data["Close"].max()
                    low_52 = data["Close"].min()

                    st.subheader(f"📈 52-Week High & Low for {stock_symbol}")
                    st.write(f"**52-Week High:** ₹{high_52:.2f}")
                    st.write(f"**52-Week Low:** ₹{low_52:.2f}")

                    # Plot
                    fig, ax = plt.subplots()
                    data["Close"].plot(ax=ax, label="Close Price", color="blue")
                    ax.axhline(high_52, color="green", linestyle="--", label="52W High")
                    ax.axhline(low_52, color="red", linestyle="--", label="52W Low")
                    ax.set_title(f"{stock_symbol} - 52 Week Stock Trend")
                    ax.set_ylabel("Price (INR)")
                    ax.legend()
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error: {e}")
