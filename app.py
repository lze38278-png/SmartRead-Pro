import streamlit as st
import os
import re
import string
import time
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
base_stop_words = set(stopwords.words('english'))

# 学术停用词表
academic_stop_words = {
    'text', 'author', 'passage', 'paragraph', 'article',
    'example', 'however', 'although', 'therefore', 'study', 'research'
}
final_stop_words = list(base_stop_words.union(academic_stop_words))


# --- 2. 核心辅助函数 ---

def process_text_for_display(text):
    """用于前端高亮展示的处理"""
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    words = nltk.word_tokenize(text)
    clean_words = []
    for word in words:
        if word not in base_stop_words and len(word) > 1 and word.isalpha():
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemma = lemmatizer.lemmatize(lemma, pos='n')
            clean_words.append(lemma)
    return set(clean_words)


def get_article_category_by_name(filename):
    """基于文件名的备用分类逻辑"""
    name_lower = filename.lower()
    if "eng1" in name_lower or "英语一" in name_lower:
        return "英语一"
    elif "eng2" in name_lower or "英语二" in name_lower:
        return "英语二"
    elif "cet4" in name_lower or "四级" in name_lower:
        return "四级"
    elif "cet6" in name_lower or "六级" in name_lower:
        return "六级"
    else:
        return "其他"


# --- 3. 数据加载 (V3.5 核心升级：递归读取) ---
@st.cache_data
def load_articles():
    articles = []
    data_folder = 'data'

    # 如果文件夹不存在，自动创建
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        return []

    # 🟢 [升级点] 使用 os.walk 遍历所有子文件夹
    for root, dirs, files in os.walk(data_folder):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                try:
                    # 尝试从文件名提取年份
                    year_match = re.search(r'20\d{2}', filename)
                    year = int(year_match.group()) if year_match else 0

                    # 🟢 [升级点] 优先用文件夹名字做分类
                    # root 是当前文件的路径，os.path.basename(root) 就是文件夹名（如 "六级"）
                    folder_name = os.path.basename(root)

                    # 如果文件直接在 data 根目录下，则尝试用文件名判断
                    if folder_name == 'data':
                        category = get_article_category_by_name(filename)
                    else:
                        category = folder_name

                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if content.strip():
                        articles.append({
                            "title": filename,
                            "year": year,
                            "category": category,
                            "content": content
                        })
                except Exception as e:
                    # 遇到编码错误或其他问题跳过
                    print(f"Skipping {filename}: {e}")
                    pass

    # 按年份倒序排列
    articles.sort(key=lambda x: x['year'], reverse=True)
    return articles


# --- 4. 界面设计 ---
st.set_page_config(page_title="SmartRead Pro V3.5", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .highlight-marker {
        background-color: rgba(255, 235, 59, 0.6);
        padding: 0 4px;
        border-radius: 4px;
        font-weight: bold;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ 智算中心")
    st.success("✅ TF-IDF 算法已调优")
    st.info("⚡ Max-DF 降噪 | 递归读取")
    st.markdown("---")

    # 1. 加载所有数据
    all_articles = load_articles()
    total_count = len(all_articles)

    if total_count > 0:
        # 获取所有年份
        years = [a['year'] for a in all_articles if a['year'] > 0]
        # 防止只有0年导致报错
        if years:
            min_y, max_y = (min(years), max(years))
        else:
            min_y, max_y = (2010, 2025)

        st.subheader("📅 语料库范围")
        selected_range = st.slider("年份筛选", min_y, max_y, (min_y, max_y))

        # 试卷类型多选框
        available_categories = sorted(list(set([a['category'] for a in all_articles])))

        # 默认全部选中
        selected_cats = st.multiselect(
            "📚 试卷类型 (可多选)",
            options=available_categories,
            default=available_categories
        )

        # 核心筛选逻辑
        filtered_articles = [
            a for a in all_articles
            if (selected_range[0] <= a['year'] <= selected_range[1]) and (a['category'] in selected_cats)
        ]

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("文章总量", total_count)
        with col_s2:
            st.metric("激活文章", len(filtered_articles))
    else:
        # 🟢 [修复点] 之前这里没定义 selected_cats，导致后续报错
        st.error("⚠️ 数据库为空")
        st.caption("请在 data 文件夹下放入 txt 真题文件")
        filtered_articles = []
        selected_range = (0, 0)
        selected_cats = []

st.title("🎓 SmartRead 考研英语智能伴读")
st.caption(
    f"V3.5 递归读取加强版 | 数据源: {selected_range[0]}-{selected_range[1]} | 类型: {', '.join(selected_cats) if selected_cats else '无'}")

col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_area("在此输入单词或长难句：", height=80,
                              placeholder="例如: First generation college students struggle with social class disadvantages...")
with col2:
    st.write("")
    st.write("")
    search_btn = st.button("🚀 向量检索", type="primary", use_container_width=True)

# --- 5. 核心：TF-IDF 匹配算法 ---
if search_btn:
    if not user_input.strip():
        st.warning("⚠️ 请输入内容！")
    elif not filtered_articles:
        st.error("❌ 当前筛选条件下无文章，请检查左侧筛选栏。")
    else:
        progress_text = "正在执行 Max-DF 降噪 | 构建加权矩阵..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.005)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(0.2)
        my_bar.empty()

        corpus = [item['content'] for item in filtered_articles]
        corpus.append(user_input)

        tfidf_vectorizer = TfidfVectorizer(
            stop_words=final_stop_words,
            max_df=0.6,
            min_df=1
        )

        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
            user_vector = tfidf_matrix[-1]
            document_vectors = tfidf_matrix[:-1]
            similarity_scores = cosine_similarity(user_vector, document_vectors).flatten()

            results = []
            user_lemmas_for_highlight = process_text_for_display(user_input)

            for idx, score in enumerate(similarity_scores):
                if score > 0.05:
                    item = filtered_articles[idx]
                    item['score'] = score
                    article_lemmas = process_text_for_display(item['content'])
                    item['matches'] = user_lemmas_for_highlight.intersection(article_lemmas)
                    results.append(item)

            results.sort(key=lambda x: x['score'], reverse=True)

            if not results:
                st.info("🤷‍♂️ 未找到语义相关的文章。")
            else:
                st.success(f"🎉 检索完成！为您推荐 **{len(results)}** 篇高相关真题")

                for idx, res in enumerate(results[:10]):
                    with st.container(border=True):
                        col_head_1, col_head_2 = st.columns([4, 1])
                        score_percent = round(res['score'] * 100, 1)

                        with col_head_1:
                            category_badge = f"【{res['category']}】"
                            st.markdown(f"### 🏆 Top {idx + 1} | {category_badge} [{res['year']}] {res['title']}")
                            match_str = ', '.join(res['matches']) if res['matches'] else "语义高度相关"
                            st.caption(f"🎯 命中关键词: {match_str}")

                        with col_head_2:
                            st.metric("相关度", f"{score_percent}%")

                        st.markdown("---")

                        display_content = res['content']
                        for match_word in res['matches']:
                            pattern = re.compile(r'\b({})\b'.format(re.escape(match_word)), re.IGNORECASE)
                            display_content = pattern.sub(
                                r'<span class="highlight-marker">\1</span>',
                                display_content
                            )
                        st.markdown(display_content, unsafe_allow_html=True)

        except ValueError:
            st.warning("⚠️ 无法构建向量空间，请尝试输入更具体的实义词。")