import os
import json
import requests
import streamlit as st
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import OpenAI
from dotenv import load_dotenv
from datetime import date, timedelta, datetime

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Streamlit page config
st.set_page_config(page_title="Financial Robo-Advisor", layout="wide", page_icon="🤖")

# Title and Header
st.title("🤖💼 Financial Robo-Advisor")
st.markdown("""
    Welcome to the **Financial Robo-Advisor**! This tool helps you gain financial insights based on real-time market data.
    Simply select your query type from the sidebar and let the AI handle the rest.
""")

# Function to get today's date
def get_today_date():
    today = date.today()
    return today.strftime("%Y-%m-%d")

#Function to get first day of this year
def get_start_of_year():
    today = date.today()
    start_of_year = date(today.year, 1, 1)
    return start_of_year.strftime("%Y-%m-%d")

# Function to retrieve data from the API endpoint with improved error handling
def retrieve_from_endpoint(url: str) -> dict:
    headers = {"Authorization": SECTORS_API_KEY}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        if response.status_code == 200 and response.content:
            try:
                data = response.json()  # Attempt to decode the JSON response
            except json.JSONDecodeError:
                st.error("Failed to decode JSON response.")
                return {"error": "Invalid JSON response"}
        else:
            st.error("Received an empty response from the API.")
            return {"error": "Empty response"}
    except requests.exceptions.HTTPError as err:
        st.error(f"HTTP error occurred: {err}")
        return {"error": f"HTTP error: {err}"}
    except requests.exceptions.RequestException as e:
        st.error(f"Request exception occurred: {e}")
        return {"error": f"Request error: {e}"}

    return json.dumps(data)

# Tools definition
@tool
def get_top_companies_by_tx_volume(start_date: str, end_date: str = None, top_n: int = 5) -> str:
    """Get top companies by transaction volume for a given date range."""
    max_attempts = 5  # Try up to 5 consecutive days
    original_start_date = start_date
    original_end_date = end_date if end_date else start_date

    for attempt in range(max_attempts):
        url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock={top_n}"
        data = retrieve_from_endpoint(url)
        
        if isinstance(data, dict) and 'error' in data:
            return json.dumps(data)
        
        # Parse the JSON string back into a dictionary
        data_dict = json.loads(data)
        
        if data_dict:  # If we have data, process it
            # Aggregate volume across all dates
            aggregated_data = {}
            for date_data in data_dict.values():
                for company in date_data:
                    symbol = company['symbol']
                    volume = company['volume']
                    if symbol in aggregated_data:
                        aggregated_data[symbol]['volume'] += volume
                    else:
                        aggregated_data[symbol] = company.copy()
            
            # Sort by total volume and get top N
            sorted_data = sorted(aggregated_data.values(), key=lambda x: x['volume'], reverse=True)[:top_n]
            
            return json.dumps({
                "data": sorted_data,
                "original_start_date": original_start_date,
                "original_end_date": original_end_date,
                "actual_start_date": start_date,
                "actual_end_date": end_date,
                "is_original_date_range": original_start_date == start_date and original_end_date == end_date,
                "attempts": attempt + 1
            })
        
        # If no data, try the next day
        start_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        if end_date:
            end_date = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            end_date = start_date
    
    return json.dumps({
        "error": "No data available for the specified date range and subsequent attempts.",
        "original_start_date": original_start_date,
        "original_end_date": original_end_date,
        "last_checked_start_date": start_date,
        "last_checked_end_date": end_date
    })
    
@tool
def get_company_overview(stock: str) -> str:
    """Get company overview for a given stock symbol."""
    url = f"https://api.sectors.app/v1/company/report/{stock}/?sections=overview"
    return retrieve_from_endpoint(url)

@tool
def get_daily_tx(stock: str, start_date: str, end_date: str) -> str:
    """Get daily transaction data including price for a stock within a given date range."""
    url = f"https://api.sectors.app/v1/daily/{stock}/?start={start_date}&end={end_date}"
    return retrieve_from_endpoint(url)

@tool
def get_performance_since_ipo(stock: str) -> str:
    """Get stock performance since IPO listing for a given stock symbol."""
    url = f"https://api.sectors.app/v1/listing-performance/{stock}/"
    return retrieve_from_endpoint(url)

@tool
def get_index(index: str) -> str:
    """Get companies by index"""
    index = index.lower()
    url = f"https://api.sectors.app/v1/index/{index}/"
    return retrieve_from_endpoint(url)

tools = [get_top_companies_by_tx_volume, get_company_overview, get_daily_tx, get_performance_since_ipo, get_index]

# LLM and Agent setup
llm = ChatGroq(
    temperature=0,
    model_name="llama3-groq-70b-8192-tool-use-preview",
    groq_api_key=GROQ_API_KEY,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
    You are the financial robo-advisor. Answer only based on the data retrieved from the APIs.
    Do not make assumptions or provide information not directly obtained from the API responses.
    Use the tools provided to fetch the necessary data, and ensure all dates are inferred or predefined correctly.

    If the query involves a specific date or relative time like 'yesterday' or 'this month',
    you must convert it to the correct date format (YYYY-MM-DD) based on today's date, which is {get_today_date()}.

    When asked about "most traded" or "top traded" stocks, always interpret this as referring to trading volume,
    and use the get_top_companies_by_tx_volume tool to retrieve this information.

    When using the get_top_companies_by_tx_volume tool:
    1. Always check the 'is_original_date' field in the response.
    2. If 'is_original_date' is false, this means the original requested date was a non-trading day.
    3. In this case, you MUST start your response by explaining that the original date was a non-trading day, and specify both the original requested date and the actual date for which data was found.
    4. Use the 'original_date' and 'actual_date' fields from the response to provide this information.
    5. After explaining the date situation, proceed to list the top companies as usual.

    If the tool returns an error message indicating no data is available, communicate this clearly to the user, mentioning both the original requested date and the last date checked.

    Always strive to provide clear and accurate information about the dates involved in the data retrieval process.

    For question related to IPO use tool get_performance_since_ipo.
    For question related to IDX or index use tool get_index.
    For question related to company comparison do not use tool get_index, use tool get_company_overview.
    For question related to close price use tool get_daily_tx, if date is null then use today's date.

    Always answer in markdown table if necessary and possible.
    """),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# General Q&A LLM setup
general_llm = OpenAI(temperature=0.7, api_key=OPENAI_API_KEY)

# Streamlit UI
st.sidebar.header("📊 Query Options")
query_option = st.sidebar.selectbox(
    "Choose a query type:",
    ["Custom Query", "Stock Performance", "Company Comparison",  "Top Companies by Volume", "IPO Performance", "Investment Comparison", "General Q&A"]
)

# Query input section
if query_option == "Custom Query":
    query = st.text_input("Enter your financial query:")
elif query_option == "Stock Performance":
    stock = st.sidebar.text_input("Enter stock symbol (e.g., BBCA):")
    start_date = st.sidebar.date_input("Start date")
    end_date = st.sidebar.date_input("End date")
    query = f"Based on the closing prices of {stock} between {start_date} and {end_date}, are we seeing an uptrend or downtrend? Try to explain why."
elif query_option == "Company Comparison":
    stock1 = st.sidebar.text_input("Enter first stock symbol:")
    stock2 = st.sidebar.text_input("Enter second stock symbol:")
    query = f"What is the company with the largest market cap between {stock1} and {stock2}? For said company, retrieve the email, phone number, listing date, and website for further research."
elif query_option == "Top Companies by Volume":
    days = st.sidebar.slider("Number of days", 1, 30, 7)
    top_n = st.sidebar.slider("Top N companies", 1, 10, 3)
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    query = f"What are the top {top_n} companies by transaction volume over the last {days} days?"
elif query_option == "IPO Performance":
    stock = st.sidebar.text_input("Enter stock symbol:")
    query = f"What is the performance of {stock} since its IPO listing?"
elif query_option == "Investment Comparison":
    stock1 = st.sidebar.text_input("Enter first stock symbol (e.g., GOTO):")
    stock2 = st.sidebar.text_input("Enter second stock symbol (e.g., BREN):")
    query = f"If I had invested into {stock1} vs {stock2} on their IPO listing dates, which one would give a better return over a 90-day horizon?"
elif query_option == "General Q&A":
    query = st.text_input("Ask any general finance-related question:")

if st.button("Get Insights"):
    if query:
        with st.spinner("Analyzing..."):
            result = agent_executor.invoke({"input": query})
            st.write("Answer:")
            st.write(result["output"])
    else:
        st.warning("Please enter a query or select query parameters.")

st.sidebar.markdown("---")
st.sidebar.info("💡This Financial Robo-Advisor uses AI to provide insights based on real-time market data. Always consult with a professional financial advisor before making investment decisions.")