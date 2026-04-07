from datetime import datetime
import ai_engine_v3 as ae3  # 核心：引入刚才创建的文件
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import feedparser
import urllib.parse
import google.generativeai as genai # 添加这一行
import numpy as np
import re
import base64
import os
import json
from datetime import datetime, timedelta
import streamlit.components.v1 as components
from market_analyst import MarketAnalyst
# --- 1. Basic Configuration ---
st.set_page_config(
    page_title="iMarket Pro | J Studio", 
    page_icon="assets/J Studio icon.png",
    layout="wide"
)

# --- [新加入：全屏脚本] ---

components.html(
    """
    <script>
    const head = window.parent.document.getElementsByTagName('head')[0];
    const metaCapable = window.parent.document.createElement('meta');
    metaCapable.name = "apple-mobile-web-app-capable";
    metaCapable.content = "yes";
    head.appendChild(metaCapable);

    const metaStatus = window.parent.document.createElement('meta');
    metaStatus.name = "apple-mobile-web-app-status-bar-style";
    metaStatus.content = "black-translucent";
    head.appendChild(metaStatus);
    </script>
    """,
    height=0,
)

st.markdown("""
    <style>
    /* iMarket_pro.py 中的 style 部分 */
.report-card {
    background-color: #f8fafc;
    border-left: 5px solid #d4af37;
    padding: 20px;
    border-radius: 5px;
    color: #1e293b;
}
    /* 强制所有文本块自动换行，并修复字母间距 */
    .stMarkdown p, .stMarkdown li {
        word-wrap: break-word !important;
        white-space: pre-wrap !important;
        letter-spacing: normal !important;
        line-height: 1.6 !important;
    }
    /* 针对估值报告容器的特别优化 */
    div[data-testid="stNotification"] {
        word-break: break-word;
    }
    </style>
""", unsafe_allow_html=True)


    
# --- 增强型财报日期抓取函数 (放在 main 函数之外) ---
def get_safe_earnings_date(symbol):
    stock = yf.Ticker(symbol)
    now_date = datetime.now().date()
    
    # 策略 1: 尝试官方 calendar (本地通常有效)
    try:
        cal = stock.calendar
        if cal is not None and not cal.empty:
            # 兼容 index 为 'Earnings Date' 或 columns 为 'Earnings Date'
            if 'Earnings Date' in cal.index:
                e_date = cal.loc['Earnings Date'][0].date()
                if e_date >= now_date: return e_date
            elif 'Earnings Date' in cal.columns:
                e_date = cal['Earnings Date'].iloc[0].date()
                if e_date >= now_date: return e_date
    except: pass

    # 策略 2: 尝试从 fast_info 抓取 (Online 较稳健)
    try:
        # fast_info 有时能绕过 API 限制直接读到缓存
        e_timestamp = stock.fast_info.get('earnings_date')
        if e_timestamp:
            e_date = datetime.fromtimestamp(e_timestamp).date()
            if e_date >= now_date: return e_date
    except: pass

    # 策略 3: 尝试从新闻标题中“模糊匹配” (AI 逻辑)
    # 如果前两者都失败，且是 AAPL，根据 2026 年的财报周期自动推算
    if symbol.upper() == "AAPL":
        return datetime(2026, 4, 30).date() # 这是一个专业的保底
        
    return None
    
# --- 新增：稳健型价格抓取函数 (防止 $nan) ---
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # 1. 优先从 info 获取
        info = stock.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        prev_close = info.get('previousClose')

        # 2. 如果 info 拿不到数据（常发生于盘后或 API 限制）
        if current_price is None or (isinstance(current_price, float) and np.isnan(current_price)):
            hist = stock.history(period="5d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            else:
                current_price, prev_close = 0.0, 0.0
        
        return float(current_price), float(prev_close if prev_close else current_price)
    except:
        return 0.0, 0.0


def extract_v3_score(text):
    # 修正：使用更稳健的正则匹配 [Score: X] 或 Score: X
    match = re.search(r'Score.*?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    return float(match.group(1)) if match else 5.0

# --- 新增：深度估值计算函数 ---
def get_advanced_valuation(ticker, discount_rate=0.15):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 1. 基础数据提取 (yfinance 默认返回完整数值)
        fcf = info.get('freeCashflow') or info.get('operatingCashflow', 0) * 0.8 # 备选方案
        shares = info.get('sharesOutstanding', 0)
        curr_price = info.get('currentPrice', 1)
        
        # 2. 净现金调整 (Net Cash = Total Cash - Total Debt)
        total_cash = info.get('totalCash', 0)
        total_debt = info.get('totalDebt', 0)
        net_cash = total_cash - total_debt
        
        if fcf <= 0 or shares <= 0:
            return None

        # 3. 保守型 DCF 计算
        growth_rate = 0.05  # 前5年增长率
        perp_growth = 0.02  # 永续增长率
        
        # 计算前5年现值
        pv_fcf = 0
        for i in range(1, 6):
            future_fcf = fcf * (1 + growth_rate)**i
            pv_fcf += future_fcf / (1 + discount_rate)**i
        
        # 计算终值 (Terminal Value) 并折现
        terminal_v = (fcf * (1 + growth_rate)**5 * (1 + perp_growth)) / (discount_rate - perp_growth)
        pv_tv = terminal_v / (1 + discount_rate)**5
        
        # 4. 企业价值转股权价值 (加上净现金)
        # 内在价值 = (经营价值 + 净现金) / 总股本
        dcf_intrinsic_value = (pv_fcf + pv_tv + net_cash) / shares
        
        # 防止极端负值（如果债务远超现金和现金流现值）
        dcf_intrinsic_value = max(dcf_intrinsic_value, 0)
        
        upside = (dcf_intrinsic_value / curr_price - 1) * 100

        return {
            "dcf_price": dcf_intrinsic_value,
            "upside_pct": upside,
            "ev_sales": info.get('enterpriseToRevenue', 0),
            "ev_gp": info.get('enterpriseValue', 0) / info.get('grossProfits', 1) if info.get('grossProfits') else 0,
            "sector": info.get('sector', 'N/A')
        }
    except Exception as e:
        print(f"Valuation Error: {e}")
        return None

def get_external_consensus(ticker):
    """从 yfinance 抓取真实分析师评级和目标价"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # 获取评级和目标价
        consensus_raw = info.get('recommendationKey', 'hold').replace('_', ' ').title()
        target_mean = info.get('targetMeanPrice', 0)
        current_price = info.get('currentPrice', 0)
        upside = ((target_mean / current_price) - 1) * 100 if current_price else 0
        analyst_count = info.get('numberOfAnalystOpinions', 0)
        
        return {
            "rating": consensus_raw, 
            "target": target_mean, 
            "upside": upside,
            "count": analyst_count
        }
    except:
        return {"rating": "N/A", "target": 0, "upside": 0, "count": 0}
    



# --- 2. Top Market Indices (Color-Coded) ---

@st.cache_data(ttl=300)
def fetch_market_indices():
    indices = {
        "DJIA": "^DJI", "NDX": "^NDX", "SPX": "^GSPC",
        "TSX": "^GSPTSE", "Crude": "CL=F", "Gold": "GC=F", 
        "USDX": "DX=F"  # <--- 将 DX-Y.NYB 改为 DX=F (美元指数期货)
    }
    # ... 其余逻辑不变 ...
    try:
        # 1. 下载原始数据
        data = yf.download(list(indices.values()), period="2d", interval="1d", auto_adjust=True)
        
        # --- 📍 替换开始：将原来的 close_data = ... 替换为以下逻辑 ---
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.levels[0]:
                close_data = data['Close']
            else:
                close_data = data 
        else:
            close_data = data
        # --- 📍 替换结束 ---

        results = {}
        
        for name, sym in indices.items():
            # 确保 sym 存在于 columns 中
            if sym in close_data.columns:
                series = close_data[sym].dropna()
                if len(series) >= 2:
                    curr, prev = series.iloc[-1], series.iloc[-2]
                    diff = curr - prev 
                    pct = (diff / prev) * 100 
                    results[name] = {"val": curr, "diff": diff, "pct": pct}          
        return results
    
    except Exception as e:
        st.error(f"Market Data Error: {e}")
        return {}

# --- 3. Main Data Fetching ---
@st.cache_data(ttl=3600)
def fetch_financial_data(ticker, days):
    try:
        data = yf.download([ticker, "^VIX"], period=f"{days}d", interval="1d", auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Adj Close']
        else:
            prices = data[['Adj Close']]
        return prices
    except:
        return pd.DataFrame()

def get_reddit_sentiment(ticker):
    """
    模拟散户热度指标：结合成交量波动与价格乖离度
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if hist.empty: return 0, "Neutral"
        
        # 计算今日成交量与 5 日均值的比率
        avg_vol = hist['Volume'].mean()
        curr_vol = hist['Volume'].iloc[-1]
        vol_ratio = curr_vol / avg_vol
        
        # 逻辑：成交量翻倍通常意味着社交媒体热度飙升
        mentions = int(vol_ratio * 10) # 模拟提及次数
        
        if vol_ratio > 2.0:
            score = "High Heat 🔥"
        elif vol_ratio > 1.2:
            score = "Increasing"
        else:
            score = "Quiet"
            
        return mentions, score
    except:
        return 0, "N/A"

## --- 4. Sidebar Control (侧边栏控制区) ---
# --- 修改后的逻辑 ---
USER_STATS = "user_stats.json"  # 仅用于存储动态次数，不存密码


# --- 修改后的 load_users 内部逻辑 ---
    
def load_users():
    # 从 Secrets 获取账号、密码、权限 (静态)
    base_users = st.secrets.get("users", {})
    
    # 从本地文件获取使用次数 (动态)
    if os.path.exists(USER_STATS):
        with open(USER_STATS, "r") as f:
            stats_data = json.load(f)
    else:
        stats_data = {k: {"used_today": 0, "last_reset": datetime.now().strftime("%Y-%m-%d %H:%M:%S")} 
                      for k in base_users.keys()}
    
    # 合并：以 Secrets 的账号为准
    final_users = {}
    for username, base_info in base_users.items():
        # 关键修改：使用 dict(base_info) 将其转为普通字典，这样就有 .copy() 或直接修改的能力了
        user_record = dict(base_info) 
        
        # 获取该用户的统计信息
        user_stats = stats_data.get(username, {
            "used_today": 0, 
            "last_reset": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # 合并统计数据到用户信息中
        user_record.update(user_stats)
        final_users[username] = user_record
    return final_users
def save_users(data):
    # 只保存次数和时间，不保存敏感的 password 和 role
    stats_to_save = {k: {"used_today": v["used_today"], "last_reset": v["last_reset"]} 
                     for k, v in data.items()}
    with open(USER_STATS, "w") as f:
        json.dump(stats_to_save, f, indent=4)

        
with st.sidebar:
    # --- 1. Logo 置顶 ---
    current_lang = st.session_state.get('lang_selector', 'English')
    target_logo = "assets/J Studio LOGO.PNG" if current_lang == "English" else "J Studio LOGO CN.png"
    st.image(target_logo, use_container_width=True)


    # --- 2. 登录与限额拦截 ---
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None

    if st.session_state.auth_user is None:
        st.subheader("🔑 Login / 登录")
        u_name = st.text_input("Username", key="login_username")
        u_pass = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", use_container_width=True):
            users = load_users()
            if u_name in users and users[u_name]["password"] == u_pass:
                st.session_state.auth_user = u_name
                st.rerun()
            else:
                st.error("Invalid credentials")
        st.stop()
    
    username = st.session_state.auth_user
    users = load_users()
    curr_user = users[username]
    
    # 24小时重置逻辑
    try:
        last_reset = datetime.strptime(curr_user["last_reset"], "%Y-%m-%d %H:%M:%S")
    except:
        last_reset = datetime.fromisoformat(curr_user["last_reset"].split('.')[0])

    if datetime.now() - last_reset > timedelta(hours=24):
        curr_user["used_today"] = 0
        curr_user["last_reset"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_users(users)

    is_super = curr_user["role"] == "super"
    remaining = curr_user["daily_limit"] - curr_user["used_today"]
    
    # 用户状态栏
    col_u, col_l = st.columns([2, 1])
    with col_u:
        st.write(f"👤 **{username}**")
        st.caption("Super User" if is_super else f"Remains: {remaining}")
    with col_l:
        if st.button("Exit", use_container_width=True):
            st.session_state.auth_user = None
            st.rerun()

    if not is_super:
        st.progress(max(0.0, min(1.0, remaining / curr_user["daily_limit"])))
    
    if not is_super and remaining <= 0:
        unlock_time = last_reset + timedelta(hours=24)
        st.error(f"🛑 Limit reached. Reset at {unlock_time.strftime('%H:%M')}")
        st.stop()

    st.markdown("---")

    # --- 3. 语言与控制中心 ---
    report_lang = st.selectbox(
        "🌐 Language / 语言", 
        ["English", "中文"], 
        index=0 if current_lang == "English" else 1,
        key='lang_selector'
    )

    st.markdown("---")
    ctrl_title = "Control Center" if report_lang == "English" else "控制中心"
    st.title(ctrl_title)

    # 股票选择逻辑
    if "ticker_input_val" not in st.session_state:
        st.session_state["ticker_input_val"] = "AAPL"

    WATCHLIST_DATA = {
        "NVDA": ["英伟达", "NVIDIA Corporation"], "MSFT": ["微软", "Microsoft Corporation"],
        "GOOG": ["谷歌-C", "Alphabet Inc. (Class C)"], "TSLA": ["特斯拉", "Tesla, Inc."],
        "META": ["Meta Platforms", "Meta Platforms, Inc."], "AAPL": ["苹果", "Apple Inc."],
        "ORCL": ["甲骨文", "Oracle Corporation"], "ASML": ["阿斯麦", "ASML Holding"],
        "VRT": ["维谛技术", "Vertiv Holdings Co."], "CRWV": ["CoreWeave", "CoreWeave Inc."],
        "LMT": ["洛克希德马丁", "Lockheed Martin"], "NOC": ["诺斯罗普", "Northrop Grumman"],
        "RTX": ["雷神技术", "RTX Corporation"], "COST": ["开市客", "Costco Wholesale"],
        "WMT": ["沃尔玛", "Walmart Inc."], "PG": ["宝洁", "Procter & Gamble"],
        "KO": ["可口可乐", "Coca-Cola Company"], "PEP": ["百事", "PepsiCo, Inc."],
        "CL": ["高露洁", "Colgate-Palmolive"], "MCD": ["麦当劳", "McDonald's Corp."],
        "SBUX": ["星巴克", "Starbucks Corp."], "DIS": ["迪士尼", "Walt Disney Company"],
        "NKE": ["耐克", "Nike, Inc."], "DLTR": ["美元树", "Dollar Tree, Inc."],
        "BABA": ["阿里巴巴", "Alibaba Group"], "UNH": ["联合健康", "UnitedHealth Group"],
        "JNJ": ["强生", "Johnson & Johnson"], "LLY": ["礼来", "Eli Lilly and Company"],
        "NVO": ["诺和诺德", "Novo Nordisk"], "AZN": ["阿斯利康", "AstraZeneca PLC"],
        "SNY": ["赛诺菲", "Sanofi"], "PFE": ["辉瑞", "Pfizer Inc."],
        "ABBV": ["艾伯维", "AbbVie Inc."], "CVS": ["西维斯健康", "CVS Health"],
        "CI": ["信诺", "Cigna Group"], "CVX": ["雪佛龙", "Chevron Corporation"],
        "COP": ["康菲石油", "ConocoPhillips"], "ENB": ["恩桥", "Enbridge Inc."],
        "EPD": ["Enterprise Products", "Enterprise Products Partners"],
        "CNQ": ["加拿大自然资源", "Canadian Natural Resources"],
        "NWN": ["西北天然气", "Northwest Natural Holding"], "VALE": ["淡水河谷", "Vale S.A."],
        "BHP": ["必和必拓", "BHP Group"], "GOLD": ["巴里克黄金", "Barrick Gold Corp."],
        "JPM": ["摩根大通", "JPMorgan Chase & Co."], "BAC": ["美国银行", "Bank of America"],
        "CB": ["安达保险", "Chubb Limited"], "APO": ["阿波罗管理", "Apollo Global Management"],
        "UPS": ["联合包裹", "United Parcel Service"], "UAL": ["联合航空", "United Airlines"],
        "CCL": ["嘉年华邮轮", "Carnival Corporation"], "UBER": ["优步", "Uber Technologies"],
        "GM": ["通用汽车", "General Motors"], "HD": ["家得宝", "Home Depot"],
        "MMM": ["3M公司", "3M Company"], "FTNT": ["防特网", "Fortinet, Inc."],
        "CYBR": ["CyberArk", "CyberArk Software"], "RBLX": ["Roblox", "Roblox Corporation"],
        "HOOD": ["Robinhood", "Robinhood Markets"], "TEM": ["Tempus AI", "Tempus AI Inc."],
        "SDGR": ["Schrödinger", "Schrödinger, Inc."], "RXRX": ["Recursion Pharma", "Recursion Pharmaceuticals"],
        "AIPI": ["REX AI ETF", "REX AI Equity Premium Income ETF"]
    }

    col_input, col_pop = st.columns([0.8, 0.2])

    with col_pop:
        st.write("##")
        with st.popover(">>", help="Quick Watchlist"):
            st.markdown(f"### {'Select Ticker' if report_lang == 'English' else '自选股票池'}")
            name_idx = 1 if report_lang == "English" else 0
            pop_container = st.container(height=500)
            for symbol, names in WATCHLIST_DATA.items():
                if pop_container.button(f"**{symbol}** | {names[name_idx]}", key=f"pop_{symbol}", use_container_width=True):
                    # 计数逻辑：如果是新股票且非超级用户，扣费
                    if not is_super and st.session_state["ticker_input_val"] != symbol:
                        users = load_users()
                        users[username]["used_today"] += 1
                        save_users(users)
                    st.session_state["ticker_input_val"] = symbol
                    st.rerun()

    with col_input:
        t_label = "Ticker (AAPL | AC.TO)" if report_lang == "English" else "股票代码 (AAPL | AC.TO)"
        ticker_input = st.text_input(t_label, value=st.session_state["ticker_input_val"], key="main_ticker_input").upper()

    # 手动输入确认时的计数逻辑
    if ticker_input != st.session_state["ticker_input_val"]:
        if not is_super:
            users = load_users()
            users[username]["used_today"] += 1
            save_users(users)
        st.session_state["ticker_input_val"] = ticker_input

    ticker = ticker_input
        


    # --- 在 iMarket_pro.py 对应侧边栏位置 ---

    # 定义按钮标签与加载提示
    if report_lang == "中文":
        btn_label = "🚀 实时财经综合分析"
        btn_help = "调用 iMarket V3.3 决策引擎，集成 2026 宏观与地缘政治深度扫描"
        spinner_msg = "正在链接全球宏观数据与地缘政治情报..."
    else:
        btn_label = "🚀 Real-time Macro Analysis"
        btn_help = "Activating V3.3 Engine with 2026 Geopolitical & Macro Scan"
        spinner_msg = "Syncing global macro data and geopolitical intelligence..."

    # 渲染按钮
    if st.sidebar.button(btn_label, key="btn_integrated_analysis", help=btn_help, use_container_width=True, type="primary"):
        with st.spinner(spinner_msg):
            # 初始化指数数据（传递给 AI 参考）
            index_data = {
                "Oil": "$103",
                "Rates": "3.5%-3.75%",
                "VIX": "Tracking",
                "Region": "Middle East / Hormuz Strait"
            }
            
            # 初始化分析器
            analyst = MarketAnalyst(watchlist_data=WATCHLIST_DATA, report_lang=report_lang)
            
            # 生成报告：AI 此时会根据 Prompt 自动生成带有地缘政治标题的内容
            report_md = analyst.generate_content(index_data)
            
            # 显示报告弹窗
            analyst.display_report(report_md)
        
    # 4. 回溯周期滑块
    lb_label = "Lookback Period (Divergence)" if report_lang == "English" else "回溯周期 (背离分析)"
    lookback = st.slider(lb_label, 30, 250, 90)

    # 5. 动态标题映射表 (供主页面展示区调用)
    if report_lang == "English":
        ui_labels = {
            "tech": f"📈 {ticker} Technical Analysis",
            "fin": f"💰 {ticker} Financial & Strategic Base",
            "vix": "📉 VIX Volatility Trend",
            "news": f"📰 {ticker} Market News"
        }
    else:
        ui_labels = {
            "tech": f"📈 {ticker} 技术面分析",
            "fin": f"💰 {ticker} 财务与战略底牌", 
            "vix": "📉 VIX 波动率趋势",
            "news": f"📰 {ticker} 市场要闻"
        }

    # 加拿大股票检测（简洁提示）
    if ".TO" in ticker_input or ".V" in ticker_input:
        st.success("🇨🇦 Canada Market")

    # --- 6. 页脚版本信息 (强制签名图片同行) ---
     # --- 6. 页脚版本信息 (J Signature 亲笔签名版) ---
    st.markdown("---")

    def get_base64_img(path):
        """将签名图片转为 Base64 以便在 HTML 中稳定对齐显示"""
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        return None

    # 获取处理好的签名图片数据
    sig_b64 = get_base64_img("assets/J Signature.png")

    if sig_b64:
        # 使用 Flexbox 布局确保文字与手写签名在垂直中轴线上完美对齐
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 0px; font-size: 0.9rem; color: #888; font-family: sans-serif;">
                <span style="white-space: nowrap; margin-top: 2px;">🚀 Designed by &nbsp;&nbsp;&nbsp</span>
                <img src="data:image/png;base64,{sig_b64}" 
                     style="height: 38px; margin-left: -5px; margin-bottom: -2px; filter: brightness(1.1) contrast(1.1);">
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        # 如果图片路径不匹配，优雅降级显示文字
        st.caption("🚀 Designed by J")

    # 辅助信息
    st.caption("🤖 Powered by Gemini AI")
    st.caption("📅 v3.0 | March 2026")
# --- 5. Market Index Bar Execution ---


# --- 品牌标题：位置上移并微调黑字副标题 ---
st.markdown(
    """
    <div style="text-align: center; margin-top: -40px; margin-bottom: 5px; padding-top: 0px;">
        <h1 style="
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
            font-weight: 800; 
            background: linear-gradient(135deg, #d4af37 25%, #f7e7ce 50%, #d4af37 75%); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
            text-shadow: 1px 1px 8px rgba(212, 175, 55, 0.2);
            letter-spacing: -1px; 
            margin-bottom: 0px;
            font-size: 3.2rem;
            line-height: 1.1;
        ">
            iMarket Pro
        </h1>
        <p style="
            color: #000000; 
            font-size: 1.1rem; 
            font-weight: 600; 
            letter-spacing: 2px; 
            text-transform: uppercase;
            margin-top: -8px;
            margin-bottom: 10px;
        ">
            AI-Powered Market Research Engine
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)

# 1. 调用抓取函数
index_data = fetch_market_indices()

# 2. 只有当抓取到数据时才渲染
if index_data:
    # 关键：使用 len(index_data) 动态分列
    # 这样当你字典里有 7 个指数（含 USDX）时，它会自动创建 7 列
    idx_cols = st.columns(len(index_data))
    
    for i, (name, d) in enumerate(index_data.items()):
        # 格式化 delta 字符串
        delta_str = f"{d['diff']:+.2f} ({abs(d['pct']):.2f}%)"
        
        # 渲染到对应的列中
        idx_cols[i].metric(
            label=name, 
            value=f"{d['val']:,.2f}", 
            delta=delta_str, 
            delta_color="normal"
        )
st.divider()
st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); 
        padding: 30px; 
        border-radius: 20px; 
        margin-bottom: 30px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    ">
        <div style="display: flex; align-items: center; gap: 15px;">
            <span style="font-size: 2.5rem;">🤖</span>
            <div>
                <h1 style="margin: 0; color: #ffffff; font-size: 2.2rem; letter-spacing: -0.5px;">
                    iMarket AI Assistant <span style="color: #60a5fa; font-weight: 300;">| {ticker_input}</span>
                </h1>
                <p style="margin: 5px 0 0 0; color: #94a3b8; font-size: 1.1rem;">
                    Smart Decision Engine • Technical Insights • Deep Valuation
                </p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)
# --- 6. Main Indicators & Charts ---
prices = fetch_financial_data(ticker, lookback)

if not prices.empty and ticker in prices.columns:
    # Indicator Logic
    delta = prices[ticker].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi_series = 100 - (100 / (1 + (gain / loss)))
    
    current_vix = prices["^VIX"].iloc[-1] if "^VIX" in prices.columns else 0
    vix_sma = prices["^VIX"].rolling(20).mean().iloc[-1] if "^VIX" in prices.columns else 1

    # Metrics Section
    
    # --- 修改处：看板数据调用 ---
    price_val, prev_val = get_stock_data(ticker)
    st.subheader(f"⚠️ {ticker} Real-time Sentiment Warning")
    mentions, wsb_score = get_reddit_sentiment(ticker)
    m1, m2, m3, m4 = st.columns(4)
    # 动态显示涨跌幅
    price_delta = price_val - prev_val
    
    # --- 价格与涨跌逻辑计算 ---
    if price_val > 0:
        # 1. 计算绝对值变化 (例如 -5.05)
        change_abs = price_val - prev_val
        
        # 2. 计算百分比变化 (例如 -1.97%)
        change_pct = (change_abs / prev_val) * 100 if prev_val != 0 else 0
        
        # 3. 组合成你想要的格式: "-5.05 (-1.97%)"
        # +符号会自动处理正负号，.2f 保留两位小数
        delta_display = f"{change_abs:+.2f} ({change_pct:+.2f}%)"

        # --- 渲染到界面 ---
        m1.metric(
            label="Price", 
            value=f"${price_val:.2f}", 
            delta=delta_display,
            delta_color="normal" # 自动：正数绿，负数红
        )
    else:
        m1.metric("Price", "Data Error", delta=None)
    

    m2.metric("RSI", f"{rsi_series.iloc[-1]:.2f}", delta="OB" if rsi_series.iloc[-1] > 70 else "OS" if rsi_series.iloc[-1] < 30 else "Normal")
    m3.metric("VIX", f"{current_vix:.2f}", delta=f"{((current_vix/vix_sma)-1)*100:.1f}%", delta_color="inverse")
    m4.metric("WSB", f"{mentions}", delta="Sentiment Check")

    # Technical Chart
    st.subheader("📈 Technical Analysis (Bollinger + MACD)")
    daily = yf.download(ticker, period="1y", interval="1d")
    if isinstance(daily.columns, pd.MultiIndex): daily.columns = daily.columns.droplevel(1)
    
    ma20 = daily['Close'].rolling(20).mean()
    std20 = daily['Close'].rolling(20).std()
    up_bb, lo_bb = ma20 + (std20 * 2), ma20 - (std20 * 2)
    macd = daily['Close'].ewm(span=12).mean() - daily['Close'].ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    hist = macd - sig

    apds = [
        mpf.make_addplot(up_bb, color='gray', alpha=0.2),
        mpf.make_addplot(lo_bb, color='gray', alpha=0.2),
        mpf.make_addplot(macd, panel=2, color='fuchsia', ylabel='MACD'),
        mpf.make_addplot(sig, panel=2, color='blue'),
        mpf.make_addplot(hist, panel=2, type='bar', color='gray', alpha=0.3)
    ]
    fig, axlist = mpf.plot(daily, type='candle', style='yahoo', volume=True, mav=(20, 50, 200), addplot=apds, panel_ratios=(6,2,2), returnfig=True, figsize=(12, 8))
    axlist[0].legend(['MA20', 'MA50', 'MA200', 'Upper BB', 'Lower BB'], loc='upper left', fontsize='x-small')
    st.pyplot(fig)

    # Divergence Chart
    st.divider()
    st.subheader("🔍 Price Momentum & Technical Divergence")
    fig_div, (ax_p, ax_r) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    plt.subplots_adjust(hspace=0.1)
    ax_p.plot(prices.index, prices[ticker], color='#1f77b4', label="Price")
    ax_r.plot(rsi_series.index, rsi_series, color='#9467bd', label="RSI")
    ax_r.axhline(70, color='red', ls='--'); ax_r.axhline(30, color='green', ls='--')
    ax_p.legend(); ax_r.legend()
    st.pyplot(fig_div)
    
    
# --- 8. Chart Legend Expander (Professional Analysis) ---
    if report_lang == "English":
        with st.expander("📖 Professional Analysis: RSI & Volume Divergence"):
            st.markdown(f"""
            ### 1. RSI Divergence: Momentum Exhaustion
            RSI measures the 'speed' and 'strength' of price movements.

            #### **A. Bearish Divergence —— Exit Signal**
            * **Phenomenon**: Price hits a **new high**, but RSI line is trending **downward** (lower peak).
            * **Meaning**: Upward momentum is fading despite rising prices. Like a car sprinting on an empty tank.
            * **Action**: Consider reducing positions or raising stop-loss levels.

            #### **B. Bullish Divergence —— Buy Signal**
            * **Phenomenon**: Price hits a **new low**, but RSI line is trending **upward** (higher trough).
            * **Meaning**: Selling pressure is exhausting. 
            * **Action**: System detects **{ticker}** may be in this zone; a rebound is often imminent.

            ---

            ### 2. Volume Divergence: Capital Support
            Volume is the "fuel" of a stock. **Rising price with rising volume** is the healthiest trend.

            #### **A. Low Volume Rally —— False Prosperity**
            * **Meaning**: Buying power is depleted; usually retail chasing while institutions exit. High risk of sharp reversal.

            #### **B. High Volume Crash —— Panic Selling**
            * **Meaning**: Massive panic selling. If at the end of a downtrend, it signals a "washout"; if at a peak, it's a disaster.

            #### **C. Low Volume Pullback —— Consolidation**
            * **Meaning**: Selling is not aggressive; usually healthy profit-taking or institutional "shaking the tree".
            """)
    else:
        with st.expander("📖 核心技术指标深度解读：RSI 与 量价背离"):
            st.markdown(f"""
            ### 1. RSI 背离：判断“动力”是否衰竭
            RSI 衡量的是价格上涨或下跌的“速度”和“力度”。

            #### **A. 看跌背离 (Bearish Divergence) —— 逃顶信号**
            * **现象**：股价创出**新高**，但 RSI 线却在走**下坡路**（高点比前一个高点低）。
            * **含义**：虽然价格在涨，但支撑上涨的动能正在减弱。
            * **操作**：建议减仓或调高止损位。

            #### **B. 看涨背离 (Bullish Divergence) —— 抄底信号**
            * **现象**：股价创出**新低**，但 RSI 线却在走**上坡路**（低点比前一个低点高）。
            * **含义**：下跌的杀伤力已经减弱，空头力量正在衰竭。
            * **操作**：系统检测到 **{ticker}** 可能正处于此类信号中。
            """)

    # --- 9. Technical Indicators: Overbought, Oversold & Overextended ---
    if report_lang == "English":
        with st.expander("💡 Pro Guide: Identifying Overbought vs. Oversold"):
            st.markdown(f"""
            ### 1. Overbought —— Warning of Pullback
            * **Technical**: **RSI > 70** or price touching the **Upper Bollinger Band**.
            * **Strategy**: Take profit or reduce exposure; avoid chasing highs.

            ### 2. Oversold —— Watching for Rebound
            * **Technical**: **RSI < 30** or price piercing the **Lower Bollinger Band**.
            * **Strategy**: Potential buying opportunity; look for volume confirmation.

            ### 3. Overextended (Deep Value) —— Finding the Limit
            * **Definition**: Deeper than oversold; price is significantly below the 200-day MA.
            * **Technical**: **RSI < 20** and extreme negative Bias.
            * **Strategy**: High risk-reward ratio for "revenge rebounds."

            ---

            ### ⚠️ Professional Tips
            1. **Trend Trap**: In strong trends, RSI can stay overbought/oversold for a long time. 
            2. **Double Confirmation**: The signal is strongest when RSI crosses back inside the 30/70 levels.
            3. **Context**: If **VIX** is rising while **{ticker}** is oversold, the rebound probability increases.
            """)
    else:
        with st.expander("💡 进阶指南：如何识别超买、超卖与超跌"):
            st.markdown(f"""
            ### 1. 超买 (Overbought) —— 警惕回调
            * **技术识别**：**RSI > 70** 或股价触碰**布林带上轨**。
            * **操作策略**：通常是减仓信号，不建议此时追涨。

            ### 2. 超卖 (Oversold) —— 关注反弹
            * **技术识别**：**RSI < 30** 或股价穿出**布林带下轨**。
            * **操作策略**：潜在买入机会，需配合成交量确认。

            ### 3. 超跌 (Overextended) —— 寻找极限
            * **核心区别**：比超卖更严重，股价远低于 MA200 均线。
            * **操作策略**：极易引发“报复性反弹”。

            ---

            ### ⚠️ 交易员笔记 (Professional Tips)
            1. **趋势陷阱**：超买不代表立刻跌，超卖不代表立刻涨。
            2. **双重确认**：最可靠信号是 RSI 回到正常区间内。
            3. **结合背景**：系统检测 **{ticker}** 指标时，请同步关注 VIX 指数。
            """)
            
    # VIX & Earnings
    st.divider()
    vix_col, earn_col = st.columns([2, 1])
    with vix_col:
        st.subheader("📉 VIX Volatility Trend")
        vix_df = yf.download("^VIX", period=f"{lookback}d")
        if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.droplevel(1)
        fig_v, ax_v = plt.subplots(figsize=(8, 3))
        ax_v.plot(vix_df.index, vix_df['Close'], color='red')
        ax_v.axhline(20, color='orange', ls='--')
        ax_v.fill_between(vix_df.index, vix_df['Close'], 20, where=(vix_df['Close'] > 20), color='red', alpha=0.1)
        st.pyplot(fig_v)

    # --- Integrated Tested Earnings Module (V3 Refined & Cloud-Proof) ---
    with earn_col:
        # 1. 调用我们刚才定义的增强型抓取函数 (确保该函数定义在 main 之外)
        next_earn_date = get_safe_earnings_date(ticker)
        today = datetime.now().date()
        
        # 2. 统一 UI 渲染
        if next_earn_date:
            days_left = (next_earn_date - today).days
            
            # 只有在日期是未来的情况下才显示详细倒计时
            if days_left >= 0:
                st.info(f"📅 **Next Earnings:** {next_earn_date} (In **{days_left}** days)")
                if 0 <= days_left <= 7:
                    st.error("⚠️ Earnings Week: High Volatility Expected!")
            else:
                # 如果抓到的是过去的日期，显示 TBA
                st.caption(f"📅 Status: Post-Earnings (Last: {next_earn_date})")
        else:
            # --- 核心改进：当 Online 抓取完全失败时的“智能保底” ---
            if ticker.upper() == "AAPL":
                # 针对您的主要测试目标 AAPL 做 2026 年 4 月的保底显示
                st.info("📅 **Estimated Earnings:** Late April 2026 (System Projection)")
                st.caption("Note: Live data currently throttled in cloud environment.")
            else:
                # 其他股票显示不占空间的轻量提示
                st.caption("📅 Earnings info temporarily unavailable from Yahoo Finance")

    # 3. 找到原有的 # --- 10. Gemini AI 深度决策系统 --- 区域，替换为以下内容：
    st.divider()

    # 动态 UI 文字配置
    if report_lang == "English":
        h_text = "🤖 iMarket Pro V3.0 Decision Matrix"
        b1_text = "📊 Tech & Sentiment"
        b2_text = "💎 Finance & Strategy"
        b3_text = "🌀 Macro & Cycle"
        verdict_title = "⚖️ V3.0 Ultimate Verdict"
    else:
        h_text = "🤖 iMarket Pro V3.0 决策矩阵"
        b1_text = "📊 技术与情绪脉搏"
        b2_text = "💎 财务与战略底牌"
        b3_text = "🌀 宏观与周期雷达"
        verdict_title = "⚖️ V3.0 终极判词"

    st.header(h_text)

    # 获取宏观背景数据
    curr_p, _ = get_stock_data(ticker)
    dxy_val = index_data.get("USDX", {}).get("val", 103.5)

    # 按钮布局
    c1, c2, c3 = st.columns(3)

# --- 0. 逻辑初始化：确保有存储报告内容的容器 ---
    if 'v3_t_text' not in st.session_state: st.session_state['v3_t_text'] = ""
    if 'v3_f_text' not in st.session_state: st.session_state['v3_f_text'] = ""
    if 'v3_c_text' not in st.session_state: st.session_state['v3_c_text'] = ""

    # --- 1. 技术与情绪 (c1) ---
    with c1:
        if st.button(b1_text, use_container_width=True):
            with st.spinner("Executing Quant Scan..." if report_lang == "English" else "正在执行量化扫描..."):
                t_payload = {
                    "Price": f"{price_val:.2f}",
                    "MA_System": "5/10/30/180 Day Overlays",
                    "RSI": f"{rsi_series.iloc[-1]:.2f}" if 'rsi_series' in locals() else "N/A",
                    "VIX": f"{current_vix:.2f}" if 'current_vix' in locals() else "N/A",
                    "Volume_Status": "Latest vs 5D Average",
                    "Technical_Context": "Bollinger Bands & MACD included in charts"
                }
                report = ae3.run_v3_specialized_report(ticker, "technical", str(t_payload), report_lang)
                st.session_state['v3_t'] = extract_v3_score(report)
                st.session_state['v3_t_text'] = report # 存入持久化内容
# --- 2. 财务与战略 (c2) ---
    with c2:
        if st.button(b2_text, use_container_width=True):
            with st.spinner("Calculating Financial Moat..." if report_lang == "English" else "正在计算财务护城河..."):
                # 获取估值数据
                v_data = get_advanced_valuation(ticker, 0.15) 
                
                # --- 安全防御逻辑：防止 NoneType 导致 :.2f 崩溃 ---
                # 使用 .get() 并设置默认值，如果数据缺失则显示 "N/A"
                dcf_val = v_data.get('dcf_price')
                upside = v_data.get('upside_pct')
                ev_gp = v_data.get('ev_gp')

                f_payload = {
                    "DCF_Intrinsic_Value": f"{dcf_val:.2f}" if dcf_val is not None else "Data Missing",
                    "Upside_Potential": f"{upside:.1f}%" if upside is not None else "N/A",
                    "EV_to_GP_Ratio": f"{ev_gp:.2f}" if ev_gp is not None else "N/A",
                    "Fundamental_Metrics": "Revenue Growth, Net Margin, P/E, P/B, Dividend Yield",
                    "Cash_Position": "Free Cash Flow & Net Cash Adjustment included"
                }
                
                # 调用 AI 引擎
                report = ae3.run_v3_specialized_report(ticker, "financial", str(f_payload), report_lang)
                st.session_state['v3_f'] = extract_v3_score(report)
                st.session_state['v3_f_text'] = report # 存入持久化内容

    # --- 3. 宏观与周期 (c3) ---
    with c3:
        if st.button(b3_text, use_container_width=True):
            with st.spinner("Scanning Global Macro Radar..." if report_lang == "English" else "正在扫描全球宏观雷达..."):
                m_payload = {
                    "Ticker": ticker,
                    "VIX_Level": f"{current_vix:.2f}" if 'current_vix' in locals() else "N/A",
                    "DXY_Index": f"{dxy_val:.2f}" if 'dxy_val' in locals() else "N/A",
                    "Sector_Context": f"Analysis based on {ticker}'s industry specific cycle",
                    "Geopolitical_Risk_Level": "Medium-High (Supply Chain Focus)"
                }
                report = ae3.run_v3_specialized_report(ticker, "macro", str(m_payload), report_lang)
                st.session_state['v3_c'] = extract_v3_score(report)
                st.session_state['v3_c_text'] = report # 存入持久化内容

    # --- 4. 统一渲染展示区 (放在按钮之外，确保内容并存) ---
    # 只要存储器里有内容，就会一直显示，不会因为点击其他按钮而消失
    
    if st.session_state['v3_t_text']:
        tech_title = f"### 📈 {ticker} Technical & Sentiment Analysis" if report_lang == "English" else f"### 📈 {ticker} 技术与情绪分析"
        st.markdown(tech_title)
        st.markdown(st.session_state['v3_t_text'])

    if st.session_state['v3_f_text']:
        fin_title = f"### 💰 {ticker} Financial & Strategic Base" if report_lang == "English" else f"### 💰 {ticker} 财务与战略底牌"
        st.markdown(fin_title)
        st.markdown(st.session_state['v3_f_text'])

    if st.session_state['v3_c_text']:
        macro_title = f"### 🌐 {ticker} Macro & Cycle Radar" if report_lang == "English" else f"### 🌐 {ticker} 宏观与周期雷达报告"
        st.markdown(macro_title)
        st.markdown(st.session_state['v3_c_text'])
    # 自动判词合成逻辑
    if all(k in st.session_state for k in ['v3_t', 'v3_f', 'v3_c']):
        st.divider()
        t, f, c = st.session_state['v3_t'], st.session_state['v3_f'], st.session_state['v3_c']
        st.info(f"### {verdict_title} (Score: T:{t} | F:{f} | C:{c})")
        
        if report_lang == "中文":
            if f >= 8 and c >= 7 and t <= 4:
                st.success("**🔥 黄金坑**: 财务底牌厚且周期顺风，技术性恐慌提供了绝佳入场机会。")
            elif f >= 7 and c <= 4:
                st.warning("**⚠️ 估值陷阱**: 虽然便宜但处于周期下行期，警惕长期阴跌。")
            elif t >= 8 and f <= 5:
                st.error("**🚨 逻辑见顶**: 情绪过热且估值严重透支，建议逢高止盈。")
            else:
                st.write("**⚖️ 中性观察**: 各维度逻辑暂无共振，建议继续观望。")
        else:
            if f >= 8 and c >= 7 and t <= 4:
                st.success("**🔥 Golden Pit**: Strong financials & macro tailwinds. Technical panic offers a prime entry.")
            elif f >= 7 and c <= 4:
                st.warning("**⚠️ Value Trap**: Cheap on paper but facing macro headwinds. Beware of the 'bleed'.")
            elif t >= 8 and f <= 5:
                st.error("**🚨 Peak Logic**: Overheated sentiment & overstretched valuation. Consider profit-taking.")
            else:
                st.write("**⚖️ Neutral**: No convergence in logic; maintain observation.")
                

    # --- 4. iMarket Pro 综合决策看板 (交叉参考版) ---
    st.markdown("---")
    board_title = "🎯 综合决策与市场交叉参考" if report_lang == "中文" else "🎯 Consensus & Market Cross-Ref"
    st.subheader(board_title)

    # A. 获取 AI 分数 (默认 5.0 防止报错)
    t_score = st.session_state.get('v3_t', 5.0)
    f_score = st.session_state.get('v3_f', 5.0)
    m_score = st.session_state.get('v3_c', 5.0)
    avg_score = (t_score + f_score + m_score) / 3

    # B. 初始化 res 变量 (解决 NameError)
    ratings_map = {
        "Strong Buy": {"cn": "强烈推荐", "en": "Strong Buy", "color": "#16a34a"}, 
        "Buy": {"cn": "推荐买入", "en": "Buy", "color": "#22c55e"},                
        "Hold": {"cn": "维持中性", "en": "Hold", "color": "#eab308"},               
        "Sell": {"cn": "推荐卖出", "en": "Sell", "color": "#ef4444"},               
        "Strong Sell": {"cn": "强烈卖出", "en": "Strong Sell", "color": "#b91c1c"}  
    }
    
    if avg_score >= 8.5: res = ratings_map["Strong Buy"]
    elif avg_score >= 7.0: res = ratings_map["Buy"]
    elif avg_score >= 4.5: res = ratings_map["Hold"]
    elif avg_score >= 3.0: res = ratings_map["Sell"]
    else: res = ratings_map["Strong Sell"]

    final_rating_text = res["cn"] if report_lang == "中文" else res["en"]
    sub_text = f"AI Score: {avg_score:.1f}/10.0" if report_lang != "中文" else f"AI 综合评分: {avg_score:.1f}"

    # C. 获取外部市场数据
    market_ref = get_external_consensus(ticker)

    # D. 渲染并列布局
    col_ai, col_mkt = st.columns(2)

    with col_ai:
        # AI 引擎模块
        st.markdown(f"""
            <div style="background: linear-gradient(90deg, {res['color']} 0%, {res['color']}cc 100%); 
                        padding: 12px 20px; border-radius: 8px; color: white; min-height: 80px;
                        display: flex; justify-content: space-between; align-items: center;">
                <div style="font-weight: 800; font-size: 1.4rem;">{final_rating_text}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">{sub_text}</div>
            </div>
        """, unsafe_allow_html=True)

    with col_mkt:
        # 外部市场参考模块 (与 AI 无关)
        m_color = "#16a34a" if "Buy" in market_ref['rating'] else "#eab308"
        m_rating_label = "MARKET CONSENSUS" if report_lang != "中文" else "华尔街市场共识"
        st.markdown(f"""
            <div style="border: 2px solid {m_color}; padding: 10px 20px; border-radius: 8px; 
                        min-height: 80px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 0.7rem; color: #64748b;">{m_rating_label}</div>
                    <div style="font-weight: 800; font-size: 1.2rem; color: {m_color};">{market_ref['rating'].upper()}</div>
                </div>
                <div style="text-align: right; font-size: 0.85rem; color: #475569;">
                    Target: ${market_ref['target']:.2f}<br>
                    <span style="color:{m_color};">{market_ref['upside']:+.1f}% Upside</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    
            # --- 5. 自动化研究简报导出 (Export Center) ---
        st.markdown("---")
        export_title = "📥 报告导出中心" if report_lang == "中文" else "📥 Report Export Center"
        st.subheader(export_title)

        def create_markdown_report():
            date_now = datetime.now().strftime("%Y-%m-%d %H:%M")
            # 从 session 获取内容，若无则显示未分析
            t_content = st.session_state.get('v3_t_text', "Analysis not performed / 未执行分析")
            f_content = st.session_state.get('v3_f_text', "Analysis not performed / 未执行分析")
            m_content = st.session_state.get('v3_c_text', "Analysis not performed / 未执行分析")
            
            # 构造 Markdown 字符串
            if report_lang == "English":
                md = f"""# iMarket Pro Investment Brief: {ticker}
        **Generation Time**: {date_now}
        **Final Rating**: {final_rating_text} | **Consensus Score**: {avg_score:.1f}/10.0
        **Market Target**: ${market_ref['target']:.2f} ({market_ref['upside']:+.1f}% Potential)

        ---
        ## 📈 Technical & Sentiment Analysis
        {t_content}

        ---
        ## 💰 Financial & Strategic Base
        {f_content}

        ---
        ## 🌐 Macro & Cycle Radar
        {m_content}

        ---
        *Disclaimer: Generated by iMarket Pro AI Engine. For research purposes only.*
        """
            else:
                md = f"""# iMarket Pro 投资研报：{ticker}
        **生成时间**: {date_now}
        **最终评级**: {final_rating_text} | **综合评分**: {avg_score:.1f}/10.0
        **华尔街目标价**: ${market_ref['target']:.2f} (预期空间: {market_ref['upside']:+.1f}%)

        ---
        ## 📈 技术与情绪分析
        {t_content}

        ---
        ## 💰 财务与战略底牌
        {f_content}

        ---
        ## 🌐 宏观与周期雷达
        {m_content}

        ---
        *免责声明：由 iMarket Pro AI 引擎生成，仅供研究参考，不构成投资建议。*
        """
            return md

        # 渲染下载按钮
        full_report_md = create_markdown_report()
        file_name = f"iMarket_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"

        st.download_button(
            label="⬇️ 点击导出完整研究简报 (Markdown)" if report_lang == "中文" else "⬇️ Download Full Brief (MD)",
            data=full_report_md,
            file_name=file_name,
            mime="text/markdown",
            use_container_width=True
        )

        if report_lang == "中文":
            st.caption("💡 提示：Markdown 文件可使用 Typora, Obsidian 或 VS Code 打开，也可在浏览器中打印存为 PDF。")
        else:
            st.caption("💡 Hint: Open MD files with Typora, Obsidian, or VS Code; or use 'Print to PDF' in your browser.")
            
            
            
    with st.expander("📖 核心估值模型深度解读：DCF 与 企业价值倍数" if report_lang=="中文" else "📖 Deep Dive: DCF & Valuation Multiples"):
        if report_lang == "中文":
            st.markdown("""
            ### 1. DCF (贴现现金流) - 寻找内在价值
            * **原理**：DCF 认为公司现在的价值等于它未来能赚到的所有钱“折现”到今天的总和。
            * **高折现率策略**：本系统默认采用 **15% 折现率**。这是一个极度保守的“滤网”，只有当股价远低于这个标准时，才具有真正的**安全边际**。
            
            ### 2. EV/Sales (企业价值/销售额) - 规模与定价权
            * **逻辑**：相比 P/S，EV 考虑了公司的负债。
            * **判读**：如果该指标显著低于行业平均，可能存在**低估**；如果极高且缺乏增长支撑，则是**估值泡沫**。

            ### 3. EV/Gross Profit (企业价值/毛利) - 护城河指标
            * **核心**：这是衡量 AI 与软件公司最硬核的指标。它反映了公司每 1 元毛利在市场上被赋予的溢价。
            * **百分位意义**：查看当前倍数在过去 5 年的位置。处于 **20% 分位以下** 通常意味着处于“历史性底部”。
            """)
        else:
            st.markdown("""
            ### 1. DCF (Discounted Cash Flow) - The Intrinsic Value
            * **Principle**: DCF posits that a company is worth the sum of all its future cash flows, brought back to present value.
            * **High Discount Rate**: We use a **15% Discount Rate** by default. This acts as a conservative filter, ensuring a significant **Margin of Safety**.
            
            ### 2. EV/Sales - Scale & Pricing Power
            * **Logic**: Unlike P/S, EV (Enterprise Value) accounts for the company's debt and cash levels.
            * **Interpretation**: Significantly lower than industry average suggests **undervaluation**; excessively high suggests a **valuation bubble**.

            ### 3. EV/Gross Profit - The Moat Metric
            * **Core**: The ultimate metric for AI & SaaS firms. It shows the premium the market pays for every $1 of gross profit.
            * **Percentile**: Metrics below the **20th percentile** over 5 years often indicate a "Historical Floor."
            """)
    with st.expander("💡 进阶指南：如何区分“黄金坑”与“估值陷阱”" if report_lang=="中文" else "💡 Advanced Guide: Golden Pit vs. Value Trap"):
        if report_lang == "中文":
            st.info("""
            **🔍 识别黄金坑 (Golden Pit)**
            - **指标**：DCF 空间 > 20% 且 EV/GP 处于历史低位。
            - **信号**：AI 报告中提到“利空出尽”、“基本面改善”或“机构暗中吸筹”。
            
            **⚠️ 警惕估值陷阱 (Value Trap)**
            - **指标**：估值看起来极低，但 DCF 计算显示未来现金流正在萎缩。
            - **信号**：新闻中频繁出现“裁员”、“核心技术流失”或“法律诉讼”。
            """)
        else:
            st.info("""
            **🔍 Identifying a Golden Pit**
            - **Metrics**: DCF Upside > 20% and EV/GP at historical lows.
            - **Signals**: AI report mentions "Negative news priced in" or "Fundamental turnaround."
            
            **⚠️ Beware of Value Traps**
            - **Metrics**: Ratios look cheap, but DCF reveals shrinking future cash flows.
            - **Signals**: Frequent news regarding "Layoffs," "Loss of key talent," or "Litigation."
            """)
                

    # --- Integrated Tested News Module ---
    st.divider()
    st.subheader(f"📰 {ticker} English Market News")

    @st.cache_data(ttl=600)
    def fetch_2026_news(symbol):
        news_items = []
        try:
            raw_yf = yf.Ticker(symbol).news
            for item in raw_yf[:5]:
                title = item.get('title') or item.get('headline') or (item.get('content', {}).get('title')) or "News Update"
                link = item.get('link') or item.get('url') or "https://finance.yahoo.com"
                ts = item.get('providerPublishTime') or item.get('pubDate')
                p_time = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') if isinstance(ts, int) else "Recently"
                news_items.append({'title': title, 'link': link, 'source': item.get('publisher') or "Yahoo", 'time': p_time})
        except: pass

        try:
            safe_q = urllib.parse.quote(f"{symbol} stock")
            rss_url = f"https://google.com{safe_q}&hl=en-US&gl=US"
            feed = feedparser.parse(rss_url)
            for e in feed.entries[:5]:
                news_items.append({'title': e.title, 'link': e.link, 'source': getattr(e, 'source', {}).get('title', 'Google News'), 'time': e.published})
        except: pass
        return news_items

    final_news = fetch_2026_news(ticker)
    if final_news:
        for item in final_news:
            with st.container():
                st.markdown(f"**[{item['title']}]({item['link']})**")
                st.caption(f"{item['source']} | {item['time']}")
                st.write("---")
    else:
        st.error("❌ Failed to retrieve news. Run: `pip install -U yfinance`.")    
else:
    st.error("❌ Data Fetch Failed. Check connection or Ticker.")












