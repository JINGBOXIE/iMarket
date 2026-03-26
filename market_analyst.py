import streamlit as st
import yfinance as yf
import ai_engine_v3 as ae3 
from datetime import datetime
import streamlit.components.v1 as components

class MarketAnalyst:
    def __init__(self, watchlist_data, report_lang="中文"):
        self.watchlist_data = watchlist_data
        self.report_lang = report_lang

    def _get_stock_data(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                curr = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = ((curr - prev) / prev) * 100
                return curr, change
        except Exception:
            pass
        return 0, 0

    def generate_content(self, index_data):
        vol_str = "市场波动平稳 (Stable Market)" 
        volatility_items = []
        for symbol, names in list(self.watchlist_data.items()):
            price, change = self._get_stock_data(symbol)
            if abs(change) >= 1.5:
                name = names[0] if self.report_lang == "中文" else names[1]
                volatility_items.append(f"{symbol}({name}): {change:+.2f}%")
        if volatility_items:
            vol_str = ", ".join(volatility_items)

        # --- 核心强化：全球宏观、地缘政治与季节性集成的 Prompt ---
        full_prompt = f"""
        Role: iMarket Pro Chief Strategist & Macro Economist.
        Current Time: 2026-03-26.
        Location: Pickering, ON, Canada.

        【核心任务：动态报告命名与季节性逻辑】
        根据当前日期（3月26日）和“4月强劲行情”的前瞻，拟定一个包含 "iMarket Pro" 和 "2026" 的专业标题。

        【重点模块 1：🌎 全球宏观与地缘政治 (Macro & Geopolitics)】
        本节目标：识别影响估值的天花板。你必须分析以下三点：
        1. 2026 宏观周期定位 (Macro Cycle)：分析当前 3.5% - 3.75% 利率环境对整体流动性的压制逻辑。
        2. 地缘政治黑天鹅 (Geopolitical Tail Risks)：
           - 重点：评估伊朗与霍尔木兹海峡局势（针对 3 月 27 日这一关键时间节点）。
           - 影响：分析原油在 $103 高位对通胀的传导，以及对加股 (TSX) 能源权重股的具体提振作用。
        3. 汇率与避险 (DXY & Safe Havens)：分析美元指数 (DXY) 对黄金 (GOLD) 及跨国科技巨头 (如 AAPL, MSFT) 海外营收折算的影响。

        【重点模块 2：📊 季节性效应 (Seasonality) 参考】
        1月: 1月效应; 2月: 获利回吐; 4月: 强劲财报/退税入场; 5-10月: Sell in May/枯水期; 9月: 最弱; 11-12月: 圣诞拉力。

        【重点模块 3：📈 投资逻辑与实战指令】
        - 结合异动标的 {vol_str} 给出分析。
        - 评估高股息策略 ($MO, $TGT, $ENB) 在当前高油价、高利率波动市中的“护城河”价值。
        - 输出包含：[Score: X.X] 综合投资建议。

        【格式要求】: 
        1. 第一行是 # [动态标题]。
        2. 使用 Markdown 格式，关键数据加粗。
        3. 数字与汉字间保持 1 个空格。
        4. 全程使用语言: {self.report_lang}。
        """

        report = ae3.run_v3_specialized_report(
            ticker="Macro_Market_Scan",
            segment="macro",
            data_payload=full_prompt,
            lang=self.report_lang
        )
        return report

    @st.dialog("iMarket Pro Analysis", width="large")
    def display_report(self, content):
        st.caption(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Pickering, ON")
        st.markdown(content)
        st.divider()
        c1, c2, c3 = st.columns(3)
        
        dl_label = "📥 下载报告" if self.report_lang == "中文" else "📥 Download"
        pr_label = "🖨️ 打印" if self.report_lang == "中文" else "🖨️ Print"
        cl_label = "❌ 关闭" if self.report_lang == "中文" else "❌ Close"

        c1.download_button(dl_label, data=content, file_name="iMarket_Report_2026.md", use_container_width=True)
        if c2.button(pr_label, use_container_width=True):
            components.html("<script>window.print();</script>", height=0)
        if c3.button(cl_label, use_container_width=True):
            st.rerun()