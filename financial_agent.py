from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.googlesearch import GoogleSearch
from dotenv import load_dotenv
load_dotenv()
import os

# creating web search agent
web_search_agent = Agent(
    name = "web search agent",
    role="search in web for the information",
    model =Groq(id ="llama-3.3-70b-versatile",api_key=os.getenv("GROQ_API_KEY")),
    tools=[DuckDuckGo()],
    instructions=["Always show sources, Display result with tables and graphs"],
    show_tool_calls=True,
    markdown=True
)

#Google search

google_agent = Agent(
    name="google search agent",
    role="You are a google search agent to grab latest news information",
    model = Groq(id="llama-3.2-90b-vision-preview",api_key=os.getenv("GROQ_API_KEY")),
    tools=[GoogleSearch()],
    description="You are a news agent that helps users find the latest news and grab latest information.",
    instructions=[
        "Given a topic by the user, respond with 5 latest news items about that topic.",
        "Search for 10 news items and select the top 4 unique items.",
        "Search in English",
        "Display the results as images and graphs"
    ],
    
    show_tool_calls=True,
    markdown=True
)

### Financial Agent

Fin_agent = Agent(
    name="Finance agent",
    model = Groq(id="llama-3.3-70b-versatile",api_key=os.getenv("GROQ_API_KEY")),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
    show_tool_calls=True,
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Format your response using markdown and use tables and appropriate graph plots to display data where possible."],
    markdown=True
)


multimodel_ai_agent = Agent(
    team=[web_search_agent,google_agent,Fin_agent],
    model = Groq(id="llama-3.3-70b-versatile",api_key=os.getenv("GROQ_API_KEY")),
    instructions=['Always show sources','Use Tables, Images and graph plots to to display result'],
    show_tool_calls=True,
    markdown=True
)

multimodel_ai_agent.print_response("Summarize analyst recommendations and share some latest news for TATA with images ",stream=True,)