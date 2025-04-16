# from dotenv import load_dotenv
# import streamlit as st
# from langchain_teddynote import logging
# import uuid
# from attr import dataclass
# from langchain_core.messages.chat import ChatMessage

# load_dotenv()

# logging.langsmith("my chatbot")

# st.title('My Chatbot')
# st.markdown("LLM에 주식분석보고서 기능을 추가한 chatbot")

# if 'messages' not in st.session_state:
#     st.session_state['messages']=[]
# with st.sidebar:

#     clear_btn = st.button('Reset')
#     selected_model = st.selectbox('model' ['gpt-4o','gpt-r0-mini'],index=0)
#     company_name=st.text_input('input company_name')
#     search_result_count=st.slidebar('result',min_value=1,max_value=10,value=3)
#     apply_btn = st.button('setting complete',type='primary')
# @dataclass
# class ChatMessageWithType:
#     chat_message:ChatMessage
#     msg_type:str
#     tool_name:str

# if clear_btn:
#     st.session_state['messages']=[]
#     st.session_state['thread_id']=uuid.uuid4()

# print_messages()

# user_input=st.chat_input('ask something')
# warning_msg=st.empty()

# if apply_btn:
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from tavily import TavilyClient

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel,Field
load_dotenv()
# Streamlit 페이지 설정
st.set_page_config(page_title="재무제표 분석", layout="wide")

# 애플리케이션 제목
st.title("📊 투자분석 보고서")
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]  # 또는 os.getenv("TAVILY_API_KEY")
client = TavilyClient(api_key=TAVILY_API_KEY)


# 사용자 입력 (주식 심볼 입력)
with st.sidebar:   
    company_name = st.text_input("company name")
    search_btn=st.button("search")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

prompt = PromptTemplate.from_template(
    "Provide only the official stock ticker symbol of {company_name}, including the exchange suffix: "
    "use '.KS' for KOSPI-listed stocks and '.KQ' for KOSDAQ-listed stocks. NO NEED any suffix if it is nasdaq or sp500 such like apple is APPL "
    "For example, 'Samsung Electronics' should be returned as '005930.KS', "
    "and 'Seojin System' should be returned as '178320.KQ'. No explanations, no extra words."
)
model=ChatOpenAI(model='gpt-4o-mini')
output_parser=StrOutputParser()
chain=prompt|model|output_parser


if company_name:
    
    ticker_symbol=chain.invoke({"company_name": company_name})
    print(ticker_symbol)

if search_btn:
# Yahoo Finance에서 데이터 가져오기
    if ticker_symbol:
        try:
            stock = yf.Ticker(ticker_symbol)
            
            # 재무제표 데이터 가져오기
            income_statement =stock.financials
            
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow

            # 분기별 데이터도 가져오기
            quarterly_income = stock.quarterly_financials
            quarterly_balance = stock.quarterly_balance_sheet
            quarterly_cashflow = stock.quarterly_cashflow

            # 빈 데이터인지 확인
            if income_statement.empty or balance_sheet.empty or cash_flow.empty:
                st.warning("해당 기업의 재무 데이터가 없습니다. 다른 티커를 입력하세요.")
            else:
                with st.expander("📆 년도별별 재무 데이터 보기"):
                # 📌 손익계산서 출력
                    st.subheader("손익계산서 (Income Statement)")
                    st.dataframe(income_statement.fillna("-"))

                # 📌 대차대조표 출력
                    st.subheader("대차대조표 (Balance Sheet)")
                    st.dataframe(balance_sheet.fillna("-"))

                # 📌 현금흐름표 출력
                    st.subheader("현금흐름표 (Cash Flow Statement)")
                    st.dataframe(cash_flow.fillna("-"))

                # 📊 순이익 추세 시각화
                    st.subheader("📈 순이익(Net Income) 추세")
                    net_income = income_statement.loc["Net Income"]
                    print(f'iloc is {income_statement.index.get_loc("Net Income")}')
                    #ttm_net_income=income_statement.iloc[net_income_index,0]
                    if net_income.isna().all():
                        st.warning("순이익(Net Income) 데이터가 없습니다.")
                    else:
                        plt.figure(figsize=(10, 4))
                        plt.plot(net_income.index, net_income.values, marker="o", linestyle="-", label="Net Income")
                        plt.xlabel("연도도")
                        plt.ylabel("순이익 ($)")
                        plt.title(f"{ticker_symbol} 순이익 추세")
                        plt.legend()
                        plt.grid(True)
                        st.pyplot(plt)

                # 📊 매출액 추세 시각화
                    st.subheader("📊 매출액(Total Revenue) 추세")
                    revenue = income_statement.loc["Total Revenue"]
                    if revenue.isna().all():
                        st.warning("매출액(Total Revenue) 데이터가 없습니다.")
                    else:
                        plt.figure(figsize=(10, 4))
                        plt.plot(revenue.index, revenue.values, color="skyblue", label="Total Revenue")
                        plt.xlabel("연도도")
                        plt.ylabel("revenue(won)")
                        plt.title(f"{company_name} revenue")
                        plt.legend()
                        plt.grid(True)
                        st.pyplot(plt)

                # 📌 분기별 데이터 탭
                with st.expander("📆 분기별 재무 데이터 보기"):
                    st.subheader("📊 분기별 손익계산서")
                    st.dataframe(quarterly_income.fillna("-"))

                    st.subheader("📊 분기별 대차대조표")
                    st.dataframe(quarterly_balance.fillna("-"))

                    st.subheader("📊 분기별 현금흐름표")
                    st.dataframe(quarterly_cashflow.fillna("-"))

        except Exception as e:
            st.error(f"데이터를 가져오는 중 오류 발생: {e}")
        
        try:
            quarterly_net_income = stock.quarterly_financials.loc["Net Income"]

    # ✅ 최근 4개 분기의 합산값 (TTM Net Income)
            net_income_index=quarterly_income.index.get_loc('Net Income')
            
            ttm_net_income=quarterly_income.iloc[net_income_index,:4].sum()
            
        except KeyError:
            print("⚠️ 'Net Income' 데이터가 존재하지 않습니다.")
        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")
        net_income = ttm_net_income  # 당기순이익
        total_asset_index = quarterly_balance.index.get_loc("Total Assets")
        print(f'iloc is {total_asset_index}')
        total_assets = quarterly_balance.iloc[total_asset_index,0]  # 총자산
        total_liabilities_index=quarterly_balance.index.get_loc("Total Liabilities Net Minority Interest")
        print(f'liability is {total_liabilities_index}')
        total_liabilities = quarterly_balance.iloc[total_liabilities_index,0]

        shareholders_equity = total_assets-total_liabilities  # 자기자본
        # 총부채
        shares_index=quarterly_balance.index.get_loc("Ordinary Shares Number")
        total_shares = quarterly_balance.iloc[shares_index,0]  # 총 발행주식 수
        stock_price = stock.history(period="1d")["Close"].iloc[-1]  # 최신 종가
        eps = net_income / total_shares  # 주당순이익 (EPS)
        print(net_income,total_assets,total_liabilities,shareholders_equity,total_shares,stock_price,eps)

        # 📜 LangChain을 활용한 투자 분석 프롬프트 설정
        prompt = PromptTemplate.from_template(
            """you are a investment analyzing expert. Given the following financial data for {ticker}, calculate key financial ratios 
            and based on the key feature on the table, analyze the company and write the report 
            and present the results in a **formatted table**. all the index and column should write in english('KOREAN') and the content of the
            analyzing should be written in KOREA

            **Financial Data**
            - Net Income: {net_income}
            - Total Assets: {total_assets}
            - Shareholder's Equity: {shareholders_equity}
            - Total Liabilities: {total_liabilities}
            - Total Shares Outstanding: {total_shares}
            - Stock Price: {stock_price}
            - Earnings Per Share (EPS): {eps}

            **Required Calculations**
            - **ROA (Return on Assets)** = (Net Income / Total Assets) * 100
            - **ROE (Return on Equity)** = (Net Income / Shareholder's Equity) * 100
            - **Debt-to-Equity Ratio** = (Total Liabilities / Shareholder's Equity) * 100
            - **PER (Price-to-Earnings Ratio)** = Stock Price / EPS
            - **PBR (Price-to-Book Ratio)** = Stock Price / (Shareholder's Equity / Total Shares Outstanding)

            do not mention calculation process or formular. Present the calculated ratios in a **Markdown table format** and include a brief investment analysis per year and searching the company name, 
            analyze the weakness and advantages in the future not based on the finacial statemtsment i gave analyzing should be written in KOREA.
            """
        )
        



        # Output parser 사용 (JSON 없이 깔끔한 표 출력)
        output_parser = StrOutputParser()

        # LangChain 체인 생성
        chain = prompt | model | output_parser

        # 모델 실행
        response = chain.invoke({
            "ticker": ticker_symbol,
            "net_income": net_income,
            "total_assets": total_assets,
            "shareholders_equity": shareholders_equity,
            "total_liabilities": total_liabilities,
            "total_shares": total_shares,
            "stock_price": stock_price,
            "eps": eps
        })
        
        combined_prompt = PromptTemplate.from_template("""
        You are a financial analyst.

        Given the company's financial report and latest future strategies from web search,
        analyze the company and describe its current health and future potential.

        Financial Report:
        {financials}

        Future Strategy (from web search):
        {future_info}

        Provide a short investor briefing that includes:
        - Financial strengths and risks
        - Strategic opportunities or challenges
        - Market trends
        - Overall investment outlook

        Answer:
        """)
        chain = combined_prompt | model |StrOutputParser()
        search_result = client.search(query=f"{company_name} future strategy and their future plans", max_results=1)
        future_info = search_result["results"][0]["content"]
        report = chain.invoke({
            "financials": response,
            "future_info": future_info
        })
        
        # 결과 출력
        st.markdown(response)
        st.mardown(report)



