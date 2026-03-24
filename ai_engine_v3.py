import google.generativeai as genai
import streamlit as st
import re
import os
import datetime
  
    # ...后续逻辑
# 启用缓存：保护付费用户配额，避免重复请求
@st.cache_data(ttl=3600, show_spinner=False)
def run_v3_specialized_report(ticker, segment, data_payload, lang="中文"):

    """
    iMarket Pro 核心 AI 决策引擎 V3.3
    集成时间锚点校准与自动化 API 配置
    """
    

    # 1. 强制注入 2026 时间锚点 (解决 2024 日期幻觉)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # 2. 自动化 API 配置
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        else:
            # 兼容本地环境变量测试
            import os
            api_key = os.getenv("GEMINI_API_KEY")
            
        if not api_key:
            return "❌ Error: No API Key found in Secrets or Environment."
            
        genai.configure(api_key=api_key)
    except Exception as e:
        return f"❌ API Configuration Failed: {str(e)}"
    
    # 3. 初始化模型 (增加模型名称降级保护)
    try:
        # 优先使用您的 V3.1 预览版
        model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")
    except Exception:
        # 如果预览版在云端未就绪，降级到稳定版 1.5-flash
        model = genai.GenerativeModel("gemini-1.5-flash")

    # 4. 构造带有“时间锁”的 System Prompt
    roles = {
        "technical": "Senior Technical Analyst & Quantitative Strategist",
        "financial": "Chief Financial Officer & Value Investment Expert",
        "macro": "Global Macro Economist & Risk Management Specialist"
    }
    
    selected_role = roles.get(segment, "Financial AI Assistant")

    # 强制 AI 认知：现在是 2026 年
    system_instruction = f"""
    You are {selected_role}. 
    Current Real-World Time: {current_time} (Year 2026).
    Your task is to analyze {ticker} based on the provided Data Context.
    CRITICAL: Ignore any pre-trained memory about 'current' dates being 2024 or 2025. 
    All analysis must be rooted in the 2026 context provided.
    Output Language: {lang}
    """

    # 5. 执行推理
    try:
        full_prompt = f"{system_instruction}\n\nData Context (JSON/Dict):\n{data_payload}\n\nPlease generate a professional, high-signal report."
        
        response = model.generate_content(full_prompt)
        
        if response and response.text:
            return response.text
        else:
            return "⚠️ AI Engine returned an empty response. Please check data payload."
            
    except Exception as e:
        # 捕捉典型的 400 或 429 错误并返回友好提示
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg:
            return "❌ API Key 效验失败，请检查 Streamlit Cloud Secrets 设置。"
        return f"❌ AI Engine Error: {error_msg}"

    # 3. 定义三大专业角色
    roles = {
        "technical": "资深量化策略师 (Senior Quant Strategist)",
        "financial": "CFO 与估值专家 (CFO & Valuation Expert)",
        "macro": "全球宏观策略分析师 (Global Macro Strategist)"
    }
    role = roles.get(segment, "资深市场研究员")

    # 4. 深度维度指令集 (整合您的详细需求)
    if segment == "technical":
        analysis_requirement = """
        执行技术指标与量价配合深度分析，必须包含：
        1. **趋势与均线系统**：分析 5/10/30/180 日均线的排列情况（多头/空头/缠绕），判断当前趋势阶段。
        2. **动能指标 (MACD & RSI)**：
           - MACD：分析金叉/死叉、柱状图伸缩及顶/底背离。
           - RSI：判断 30/70 区间，识别“超买、超卖、超跌”状态。
        3. **量价配合 (Volume Profile)**：分析价格变动是否得到成交量支持（如：缩量上涨或放量下跌的逻辑）。
        4. **市场情绪 (Sentiment)**：结合 VIX 指数和情绪水位评估当前机构的情绪分布。
        5. **执行策略 (Action Plan)**：给出具体的执行建议（左侧买入、右侧加仓或防守性止盈）。
        """
    elif segment == "financial":
        analysis_requirement = """
        执行首席财务官 (CFO) 级别的深度评估，必须包含：
        1. **核心财务指标**：分析营收与利润增长质量 (YoY)、P/E (市盈率)、P/B (市净率) 及分红可靠性。
        2. **现金流与安全边际**：重点分析自由现金流 (FCF) 状况，并基于 DCF 模型评估内在价值风险。
        3. **EV/GP 护城河分析**：评估企业价值与毛利的比例，判断其在行业中的竞争优势及估值水位。
        4. **未来成长与风险**：识别潜在增长失速点、债务杠杆压力或资本支出 (CapEx) 异常。
        5. **执行策略建议**：基于估值溢价/折价情况，给出“分批布局”或“等待回调”的逻辑。
        """
    elif segment == "macro":
        analysis_requirement = """
        执行全球宏观策略分析师级别的深度评估，必须包含：
        1. **全球宏观周期 (Macro Overlay)**：分析宏观环境，结合 VIX 和 DXY (美元指数) 走势说明对估值的压制或释放逻辑。
        2. **行业周期与定位**：分析行业景气循环阶段（复苏/过热/衰退/滞胀），说明该股的“护城河”。
        3. **地缘政治与黑天鹅**：识别地缘政治、供应链风险或政策监管带来的机会与风险点。
        4. **竞对横向扫描 (Peer Comparison)**：对比主要竞争对手，分析资本效率、技术迭代或市场份额差异。
        5. **执行策略建议**：基于大周期位置（如：周期顶部减仓、衰退末期埋伏），给出资产配置建议。
        """
    else:
        analysis_requirement = "执行该维度的全方位专业市场分析。"

    # 5. 构造高强度 Prompt (强制执行格式规范)
    prompt = f"""
    Role: {role}
    Ticker: {ticker}
    Analysis Segment: {segment}
    Data Context: {data_payload}
    
    【强制任务】
    {analysis_requirement}
    
    【格式与语言要求】
    1. 必须全程使用 {lang} 输出。
    2. 采用结构化 Markdown，关键指标和最终结论必须 **加粗**。
    3. 数字与汉字之间必须保持 **1 个空格** 的距离（如：上涨 20%）。
    4. 每一项分析必须基于数据，严禁空话。
    5. 报告结尾必须包含：[Score: X.X] (X.X 为 1-10 的评分)。
    """

    # 6. 执行请求与渲染优化
    try:
        response = model.generate_content(prompt)
        # 优化换行逻辑，增强 Markdown 在 Streamlit 上的阅读体验
        formatted_text = response.text.replace("\n*", "\n\n*").replace("\n-", "\n\n-")
        return formatted_text
    except Exception as e:
        return f"❌ AI Engine 报告生成失败: {str(e)}"
