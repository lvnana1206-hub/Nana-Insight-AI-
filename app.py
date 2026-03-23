# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:04:49 2026

@author: lnn12
"""

import streamlit as st
import pandas as pd
import jieba
import jieba.analyse
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ================= 1. 页面配置 (SaaS 高级感) =================
st.set_page_config(page_title="Nana Insight AI | Global Edition", layout="wide", page_icon="📈")

# 自定义 CSS 提升视觉高级感
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stMetric { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #000000; color: white; }
    .stButton>button:hover { background-color: #333333; border: 1px solid #000000; }
    .nana-header { color: #0047AB; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; }
    .insight-card { background-color: #f0f7ff; padding: 15px; border-left: 5px solid #0047AB; border-radius: 5px; margin-bottom: 10px; }
    #wc-container { background-color: #ffffff; padding: 10px; border-radius: 10px; border: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. 核心 AI 逻辑类 =================
class NanaGlobalEngine:
    def __init__(self):
        # 扩充繁简及多语种词库
        self.neg_keywords = [
            '不給力', '氣死', '沒網路', '閃退', '關不掉', '消失', '亂發', 
            '一幫賊', '不負責', '失望', 'bug', 'error', '慢', '無法', '差'
        ]
        self.pos_keywords = [
            '好吃', '讚', '收藏', '好用', '優秀', '推薦', '感人', '給力', '解決'
        ]

    def analyze_sentiment(self, text):
        if not isinstance(text, str): return "中性"
        text = text.lower()
        score = sum([1 for w in self.pos_keywords if w in text]) - \
                sum([1 for w in self.neg_keywords if w in text])
        if score > 0: return "正面"
        elif score < 0: return "负面"
        return "中性"

    def get_keywords(self, text_list):
        full_text = " ".join([str(i) for i in text_list if pd.notnull(i)])
        # TF-IDF 提取 Top 30，为词云提供丰富数据
        tags = jieba.analyse.extract_tags(full_text, topK=30, withWeight=True)
        return pd.DataFrame(tags, columns=['标签', '权重'])
    
    def get_auto_suggestions(self, kw_df):
        # 建立 PM 专家建议库 (Key 为关键词，Value 为建议)
        knowledge_base = {
            '網路': '优化跨域网络握手逻辑，增加 CDN 加速节点。',
            '閃退': '紧急排查 iPad/特定机型适配 Bug，进行内存压力测试。',
            '配送': '建议优化配送调度算法，并增加实时配送时长预估。',
            '客服': '引入 AI 客服初筛工单，高频技术问题直接同步研发侧。',
            '收藏': '用户留存意愿强，建议增加收藏夹分类与一键分享功能。',
            '通知': '重构 Push 推送频率分级，增加用户自定义通知开关。'
        }
        
        suggestions = []
        top_tags = kw_df['标签'].tolist()[:5] # 取前 5 个关键词
        
        for tag in top_tags:
            for key, advice in knowledge_base.items():
                if key in tag:
                    suggestions.append({"问题": tag, "建议": advice})
        return suggestions

# ================= 3. UI 界面布局 =================

# --- 侧边栏 ---
with st.sidebar:
    st.markdown("<h2 class='nana-header'>Nana Insight</h2>", unsafe_allow_html=True)
    st.write("🌍 **多语种用户洞察引擎**")
    st.divider()
    uploaded_file = st.file_uploader("上传您的评论 CSV 文件", type=["csv"])
    st.info("💡 提示：系统已自动适配您的【评论、评分、地区】等字段。")
    st.markdown("<br><br><p style='color:#ccc'>Created by Nana @ SJTU</p>", unsafe_allow_html=True)

# --- 主页面内容 ---
st.markdown("<h1 class='nana-header'>Nana Insight AI: 全球用户声音分析看板</h1>", unsafe_allow_html=True)

if uploaded_file is not None:
    # 自动识别编码读取
    df = None
    for enc in ['utf-8', 'gbk', 'utf-8-sig']:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc)
            break
        except: continue

    if df is not None:
        # --- 数据预处理：模糊匹配列名 ---
        df.columns = [str(c).strip() for c in df.columns]
        
        # 定义关键词
        comment_keys = ['评论', '内容', 'comment', 'content', 'text', 'review']
        rating_keys = ['评分', 'rating', 'score', '星级']
        region_keys = ['地区', 'region', 'location', '国家']

        def find_col(keys):
            for col in df.columns:
                if any(k in col.lower() for k in keys):
                    return col
            return None

        target_comment = find_col(comment_keys)
        target_rating = find_col(rating_keys)
        target_region = find_col(region_keys)

        if target_comment:
            df = df.rename(columns={target_comment: 'comment'})
            st.success(f"✅ 已成功识别分析字段：{target_comment}")
        else:
            st.error(f"❌ 找不到评论列！检测到的列名有：{list(df.columns)}")
            st.stop()

        if target_rating: df = df.rename(columns={target_rating: 'rating'})
        if target_region: df = df.rename(columns={target_region: 'region'})

        # 启动引擎分析情緒
        engine = NanaGlobalEngine()
        df['sentiment'] = df['comment'].apply(engine.analyze_sentiment)

        # --- 第一排：核心指标卡片 ---
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("分析样本数", len(df))
        with c2:
            neg_count = len(df[df['sentiment'] == '负面'])
            st.metric("负面情绪", neg_count, delta=f"{int(neg_count/len(df)*100)}%", delta_color="inverse")
        with c3:
            avg_rating = pd.to_numeric(df['rating'], errors='coerce').mean() if 'rating' in df.columns else 0
            st.metric("平均评分", round(avg_rating, 1))
        with c4:
            st.metric("覆盖地区数", df['region'].nunique() if 'region' in df.columns else 1)

        st.divider()

        # --- 第二排：全局可视化 (情绪 + 地区) ---
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.subheader("🎭 情绪健康度分布")
            fig_pie = px.pie(df, names='sentiment', hole=0.5,
                             color='sentiment',
                             color_discrete_map={'正面':'#0047AB', '中性':'#E5E5E5', '负面':'#000000'})
            fig_pie.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_r:
            st.subheader("🗺 地区活跃度 (Region)")
            if 'region' in df.columns:
                region_counts = df['region'].value_counts().reset_index()
                fig_reg = px.bar(region_counts, x='region', y='count', color_discrete_sequence=['#0047AB'])
                st.plotly_chart(fig_reg, use_container_width=True)
            else:
                st.info("数据中未检测到有效的‘地区’字段。")

        st.divider()

        # --- 第三排：词云 (Word Cloud) - 占据整行，极具视觉冲击力 ---
        st.subheader("☁️ 用户关注热点词云 (NLP)")
        
        # 提取词频字典
        kw_df = engine.get_keywords(df['comment'].tolist())
        word_freq = dict(zip(kw_df['标签'], kw_df['权重']))
        
        if word_freq:
            with st.container():
                st.markdown("<div id='wc-container'>", unsafe_allow_html=True)
                
                # 配置词云 (这里使用了黑蓝调风格以适配你的 UI)
                # 注意：如果是 Windows 系统，通常需要指定中文字体路径，否则会乱码
                font_path = "C:/Windows/Fonts/msyh.ttc" # 微软雅黑
                
                wc = WordCloud(
                    font_path=font_path,
                    background_color="white",
                    width=1200, # 宽度加大，占据整行
                    height=400,
                    colormap="Blues",  # 使用蓝色系
                    max_words=100,
                    relative_scaling=0.5
                ).generate_from_frequencies(word_freq)

                # 在 Streamlit 中显示
                fig_wc, ax = plt.subplots(figsize=(12, 4))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                plt.tight_layout(pad=0)
                st.pyplot(fig_wc)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.write("暂无足够数据生成词云")

        st.divider()
        
        # --- 新增模块：AI 自动化决策引擎 ---
        st.subheader("🤖 AI 自动化产品建议 (Beta)")
        auto_advices = engine.get_auto_suggestions(kw_df)
        
        if auto_advices:
            # 用美观的表格展示
            advice_table = pd.DataFrame(auto_advices)
            st.table(advice_table) 
        else:
            st.info("💡 当前数据特征暂未触发现成建议，建议通过词云手动分析。")

        # --- 第四排：深度洞察建议 ---
        st.subheader("💡 Nana's AI 产品决策建议")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class='insight-card'>
                <strong>⚠️ 紧迫问题：网络与稳定性 (iPad/Web)</strong><br>
                用户反馈“没网络”、“闪退”等。建议排查 iPad 端 Bug 及海外链路适配。
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class='insight-card'>
                <strong>🔔 体验优化：通知降噪</strong><br>
                “乱发通知且关不掉”干扰使用。建议重构推送分层逻辑。
            </div>
            """, unsafe_allow_html=True)

        with st.expander("📄 查看完整原始数据"):
            st.write(df)

else:
    #引导页
    st.image("https://img.icons8.com/clouds/200/000000/data-configuration.png", width=150)
    st.info("👋 你好 Nana！请上传您的多语种评论 CSV，我将为您深度剖析产品痛点。")
    st.markdown("""
    **如何准备您的 CSV 文件？**
    1. 确保包含【评论】列（系统支持“评论内容”、“comment”等模糊匹配）。
    2. 可选包含【评分】、【地区】、【标题】。
    3. 支持繁体、简体、英文及韩语。
    """)