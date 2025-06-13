import os
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

# 🔐 Set Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCEqjX9VInQKeSlFCOXeP653_2FNWL5CW4"

# 🌟 Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# 🔍 Search tool
search_tool = DuckDuckGoSearchResults()

# 🧠 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial assistant. Use the search results to find the most recent available stock price for the company."),
    ("user", "Here are the search results for '{stock_name}':\n\n{search_results}\n\nWhat is the most recently available stock price?")
])

# 🔗 Chain
chain: Runnable = (
    {
        "stock_name": lambda x: x["stock_name"],
        "search_results": lambda x: search_tool.run(x["stock_name"] + " share price India")
    }
    | prompt
    | llm
)

# 🎨 Streamlit UI
st.set_page_config(page_title="📊 Indian Stock Info", layout="centered")
st.title("📈 Indian Stock Price & 52-Week Trend")

st.markdown("Enter a company name (e.g., **Infosys**) and its NSE symbol (e.g., **INFY.NS**) to view stock info and 1-year price trend.")

stock_name = st.text_input("Enter Stock Name")
stock_symbol = st.text_input("Enter NSE Symbol (e.g., INFY.NS, RELIANCE.NS)", help="Use Yahoo Finance NSE code (e.g., TCS.NS, HDFCBANK.NS)")

if st.button("Get Stock Info"):
    if not stock_name or not stock_symbol:
        st.warning("Please enter both stock name and NSE symbol.")
    else:
        try:
            # 🔹 Gemini summary
            with st.spinner("Getting stock summary..."):
                result = chain.invoke({"stock_name": stock_name})
                st.success("✅ Summary retrieved!")
                st.markdown(f"**Gemini Summary:**\n\n{result.content}")
                st.markdown("*📌 Note: This is not guaranteed to be real-time. For official info, check NSE/BSE.*")

            # 🔹 YFinance chart
            with st.spinner("Fetching 1-year price data..."):
                data = yf.download(stock_symbol, period="1y")
                if data.empty:
                    st.warning("⚠️ Could not fetch stock data. Check symbol or try another.")
                else:
                    # Fix: Convert Series to float
                    high_52 = float(data["Close"].max())
                    low_52 = float(data["Close"].min())

                    st.subheader(f"📊 52-Week High & Low for {stock_symbol}")
                    st.write(f"**52-Week High:** ₹{high_52:.2f}")
                    st.write(f"**52-Week Low:** ₹{low_52:.2f}")

                    # Plot
                    fig, ax = plt.subplots()
                    data["Close"].plot(ax=ax, label="Close Price", color="blue")
                    ax.axhline(high_52, color="green", linestyle="--", label="52W High")
                    ax.axhline(low_52, color="red", linestyle="--", label="52W Low")
                    ax.set_title(f"{stock_symbol} - 52 Week Price Trend")
                    ax.set_ylabel("Price (INR)")
                    ax.legend()
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error: {e}")
