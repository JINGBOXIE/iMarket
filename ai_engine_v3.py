import google.generativeai as genai
import streamlit as st
import re

# 启用缓存
@st.cache_data(ttl=3600, show_spinner=False)
def run_v3_specialized_report(ticker, segment, data_payload, lang="中文"):
    # 1. 配置 API Key
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # 2. 初始化模型
    model = genai.GenerativeModel("gemini-3.1-flash-lite-preview") 

    # 3. 定义角色
    roles = {
        "technical": "Senior Quant Strategist",
        "financial": "CFO & Valuation Expert",
        "macro": "Global Macro Strategist"
    }
    role = roles.get(segment, "Market Expert")

    # 4. 根据维度定制指令
    if segment == "technical":
        analysis_requirement = """
        分析要求：
        - 趋势：5/10/30/180 日均线排列。
        - 动能：MACD 背离与 RSI 超买/超卖/超跌。
        - 量价：成交量支撑分析。
        - 策略：给出具体的左侧/右侧执行建议。
        """
    elif segment == "financial":
        analysis_requirement = """
        分析要求：
        - 核心：营收/利润增长，P/E, P/B, 分红。
        - 估值：DCF 模型与 EV/GP 护城河分析。
        - 风险：现金流安全边际与未来增长压力。
        - 策略：基于估值溢价/折价的建议。
        """
    elif segment == "macro":
        analysis_requirement = """
        分析要求：
        - 宏观：结合 VIX 和 DXY (美元指数) 分析。
        - 周期：行业生命周期定位。
        - 地缘：供应链机会与风险。
        - 竞对：与主要对手的横向比较。
        - 策略：大周期资产配置建议。
        """
    else:
        analysis_requirement = "执行全方位专业分析。"

    # 5. 构造最终 Prompt (确保所有特殊字符都在字符串内部)
    prompt = f"""
    Role: {role}
    Ticker: {ticker}
    Analysis Segment: {segment}
    Data Context: {data_payload}
    
    【核心任务 / Core Task】
    {analysis_requirement}
    
    【评级标准 / Rating Standard】
    请根据分析结果在报告结尾给出以下等级之一：
    - 强烈推荐 (Strong Buy): Score 8.5-10
    - 推荐买入 (Buy): Score 7.0-8.4
    - 维持中性 (Hold): Score 4.5-6.9
    - 推荐卖出 (Sell): Score 3.0-4.4
    - 强烈卖出 (Strong Sell): Score 0-2.9
    
    【强制要求 / Constraints】
    1. 必须全程使用 {lang} 输出。
    2. 数字与汉字之间必须保留 1 个空格。
    3. 结尾必须包含：[Score: X] (X 为 0-10 的数字)。
    """

    # 6. 执行请求
    try:
        response = model.generate_content(prompt)
        # 格式化换行
        formatted_text = response.text.replace("\n*", "\n\n*").replace("\n-", "\n\n-")
        return formatted_text
    except Exception as e:
        return f"❌ AI Engine Error: {str(e)}"