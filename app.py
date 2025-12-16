import streamlit as st
import os
import re
import string
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# --- 1. 初始化 NLP 引擎 ---
@st.cache_resource
def download_nltk_data():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
    for r in resources:
        try:
            nltk.data.find(f'tokenizers/{r}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{r}')
            except LookupError:
                nltk.download(r)


download_nltk_data()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# --- 2. 核心 NLP 算法 (已修复数字问题) ---
def process_text(text):
    text = text.lower()
    # 去除标点
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    words = nltk.word_tokenize(text)

    clean_words = []
    for word in words:
        # 🟢 修复核心：增加 word.isalpha() 判断
        # 含义：只有当单词完全由字母组成时才保留 (过滤掉 "24", "100%", "2015" 等)
        if word not in stop_words and len(word) > 1 and word.isalpha():
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemma = lemmatizer.lemmatize(lemma, pos='n')
            clean_words.append(lemma)

    return set(clean_words)


# --- 3. 数据加载 ---
@st.cache_data
def load_articles():
    articles = []
    data_folder = 'data'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        return []

    files = os.listdir(data_folder)
    files.sort(reverse=True)

    for filename in files:
        if filename.endswith(".txt"):
            file_path = os.path.join(data_folder, filename)
            try:
                year_match = re.search(r'20\d{2}', filename)
                year = int(year_match.group()) if year_match else 0
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        articles.append({
                            "title": filename,
                            "year": year,
                            "content": content
                        })
            except Exception as e:
                pass
    return articles


# --- 4. 界面设计 (V0.2.7 风格) ---
st.set_page_config(page_title="SmartRead Pro", page_icon="🎓", layout="wide")

# 样式微调
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* 荧光笔高亮样式 */
    .highlight-marker {
        background-color: rgba(255, 235, 59, 0.6); 
        padding: 0 4px;
        border-radius: 4px;
        font-weight: bold;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.title("⚙️ 控制台")
    st.success("✅ NLP 核心已就绪")
    st.markdown("---")

    all_articles = load_articles()
    total_count = len(all_articles)

    if total_count > 0:
        years = [a['year'] for a in all_articles if a['year'] > 0]
        min_y, max_y = (min(years), max(years)) if years else (2010, 2025)

        st.subheader("📅 数据透视")
        selected_range = st.slider("年份范围筛选", min_y, max_y, (min_y, max_y))

        filtered_articles = [a for a in all_articles if selected_range[0] <= a['year'] <= selected_range[1]]

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("文章总数", total_count)
        with col_s2:
            st.metric("当前选中", len(filtered_articles))
    else:
        st.error("⚠️ 数据库为空")
        filtered_articles = []
        selected_range = (0, 0)

# 主界面标题 (回归白色大字)
st.title("🎓 SmartRead 考研英语智能伴读")
st.caption(f"V0.2.8 算法修复版 | 赋能你的每一分钟复习 | 数据源: {selected_range[0]}-{selected_range[1]}")

# 输入区
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_area("在此输入你背的单词或长难句：", height=80,
                              placeholder="试着输入: The economic growth rate involves inflation...")
with col2:
    st.write("")
    st.write("")
    search_btn = st.button("🚀 深度匹配", type="primary", use_container_width=True)

# --- 5. 匹配逻辑 ---
if search_btn:
    if not user_input.strip():
        st.warning("⚠️ 请先输入内容！")
    elif not filtered_articles:
        st.error("❌ 没有数据可供检索。")
    else:
        # 假装思考的进度条
        progress_text = "正在去除停用词、词形还原、过滤非核心数字..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.005)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(0.2)
        my_bar.empty()

        # 核心：处理用户输入 (此时数字会被过滤掉)
        user_lemmas = process_text(user_input)

        with st.expander("🧠 点击查看 NLP 语义分析内核 (已过滤数字干扰)", expanded=True):
            st.write("原始输入:", user_input)
            # 这里显示的集合里，绝对不会再有 '24' 了
            st.code(f"核心词根提取 (Set): {user_lemmas}", language="python")

        if not user_lemmas:
            st.warning("输入内容无效（可能是停用词或纯数字），请输入实义词。")
        else:
            results = []
            for item in filtered_articles:
                article_lemmas = process_text(item['content'])
                common = user_lemmas.intersection(article_lemmas)
                score = len(common)
                if score > 0:
                    item['score'] = score
                    item['matches'] = common
                    results.append(item)

            results.sort(key=lambda x: x['score'], reverse=True)

            if not results:
                st.info("🤷‍♂️ 未找到匹配文章。")
            else:
                st.success(f"🎉 检索完成！为您推荐 **{len(results)}** 篇高相关真题")

                for idx, res in enumerate(results[:10]):
                    # 卡片容器
                    with st.container(border=True):
                        # 完美的标题布局：排名 + 年份 + 标题
                        col_head_1, col_head_2 = st.columns([4, 1])

                        with col_head_1:
                            st.markdown(f"### 🏆 Top {idx + 1} | [{res['year']}] {res['title']}")
                            st.caption(f"🎯 命中关键词: {', '.join(res['matches'])}")

                        with col_head_2:
                            st.metric("匹配热度", res['score'])

                        st.markdown("---")

                        display_content = res['content']
                        for match_word in res['matches']:
                            # 正则全词匹配高亮
                            pattern = re.compile(r'\b({})\b'.format(re.escape(match_word)), re.IGNORECASE)
                            display_content = pattern.sub(
                                r'<span class="highlight-marker">\1</span>',
                                display_content
                            )

                        st.markdown(display_content, unsafe_allow_html=True)