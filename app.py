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
# [V3.6内核] 谷歌翻译
from deep_translator import GoogleTranslator


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

academic_stop_words = {
    'text', 'author', 'passage', 'paragraph', 'article',
    'example', 'however', 'although', 'therefore', 'study', 'research'
}
final_stop_words = list(base_stop_words.union(academic_stop_words))


# --- 2. 核心辅助函数 ---
def process_text_for_display(text):
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


# [V3.6内核] 智能分片翻译
@st.cache_data(show_spinner=False)
def translate_text(text):
    try:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < 1000:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk)
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk)

        full_translation = ""
        translator = GoogleTranslator(source='auto', target='zh-CN')

        for chunk in chunks:
            if chunk.strip():
                time.sleep(0.2)
                trans = translator.translate(chunk)
                if trans:
                    full_translation += trans + " "
        return full_translation
    except Exception as e:
        return f"翻译异常: {str(e)}"


# 🟢 [V3.8 新增] SmartBridge 解析器
def parse_vocabulary_paste(text):
    """
    智能解析剪贴板内容
    支持格式：
    1. apple n. 苹果
    2. banana [音标] 香蕉
    3. 纯单词列表
    """
    vocab_set = set()
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue

        # 策略A：尝试匹配行首的纯英文单词
        # 排除像 'a', 'I' 这种过短的词，除非明确就是个单词行
        match = re.match(r'^[a-zA-Z\-\']{2,}', line)
        if match:
            word = match.group()
            # 再次清洗，去掉非字母字符
            clean_word = re.sub(r'[^a-zA-Z\-]', '', word).lower()
            if clean_word not in base_stop_words:
                vocab_set.add(clean_word)

    return list(vocab_set)


# --- 3. 数据加载 ---
@st.cache_data
def load_articles():
    articles = []
    data_folder = 'data'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        return []

    for root, dirs, files in os.walk(data_folder):
        for filename in files:
            if filename.endswith(".txt") or filename.endswith(".json"):  # 预留json接口
                file_path = os.path.join(root, filename)
                try:
                    year_match = re.search(r'20\d{2}', filename)
                    year = int(year_match.group()) if year_match else 0
                    folder_name = os.path.basename(root)

                    category = get_article_category_by_name(filename) if folder_name == 'data' else folder_name

                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if content.strip():
                        # 预处理引理，加速后续匹配
                        lemmas = process_text_for_display(content)
                        articles.append({
                            "title": filename,
                            "year": year,
                            "category": category,
                            "content": content,
                            "lemmas": lemmas  # 缓存引理集合
                        })
                except Exception:
                    pass
    articles.sort(key=lambda x: x['year'], reverse=True)
    return articles


# --- 4. 界面设计 ---
st.set_page_config(page_title="SmartRead V3.8", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    div.stButton > button {
        width: 100%;
        min-height: 50px;
        font-size: 18px !important;
        border-radius: 10px;
    }
    .stTextArea textarea {
        font-size: 16px !important;
    }
    .highlight-marker {
        background-color: #ffeb3b;
        padding: 0 4px;
        border-radius: 4px;
        font-weight: bold;
        color: #000;
    }
    .vocab-badge {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.9em;
        margin-right: 5px;
        display: inline-block;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 SmartRead 考研英语智能伴读")

# 数据加载
all_articles = load_articles()
total_count = len(all_articles)

# 筛选器 (Expander)
if total_count > 0:
    years = [a['year'] for a in all_articles if a['year'] > 0]
    min_y, max_y = (min(years), max(years)) if years else (2010, 2025)
    available_categories = sorted(list(set([a['category'] for a in all_articles])))

    with st.expander("⚙️ 题库筛选设置 (点击展开)", expanded=False):
        selected_cats = st.multiselect("📚 试卷类型:", available_categories, default=available_categories)
        selected_range = st.slider("📅 年份范围", min_y, max_y, (min_y, max_y))

        filtered_articles = [
            a for a in all_articles
            if (selected_range[0] <= a['year'] <= selected_range[1]) and (a['category'] in selected_cats)
        ]
        st.caption(f"当前激活文章库: {len(filtered_articles)} 篇")
else:
    filtered_articles = []
    st.error("⚠️ 数据库为空，请检查 data 文件夹")

# ==========================================
# 🟢 V3.8 核心升级：双标签页架构
# ==========================================
tab1, tab2 = st.tabs(["🔍 查词与研读", "📥 导入生词本 (SmartBridge)"])

# --- TAB 1: 原有的查词功能 ---
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_area("输入单词或长难句：", height=100, placeholder="例如: artificial intelligence...",
                                  key="search_box")
    with col2:
        st.write("")
        st.write("")
        search_btn = st.button("🚀 向量检索", type="primary", key="btn_search")

    if search_btn and user_input.strip() and filtered_articles:
        # TF-IDF 逻辑 (保持不变)
        progress_text = "SmartRead 正在检索..."
        my_bar = st.progress(0, text=progress_text)
        time.sleep(0.1)
        my_bar.empty()

        corpus = [item['content'] for item in filtered_articles]
        corpus.append(user_input)

        try:
            tfidf_vectorizer = TfidfVectorizer(stop_words=final_stop_words, max_df=0.6, min_df=1)
            tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
            similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

            results = []
            user_lemmas = process_text_for_display(user_input)

            for idx, score in enumerate(similarity_scores):
                if score > 0.05:
                    item = filtered_articles[idx]
                    item['score'] = score
                    item['matches'] = user_lemmas.intersection(item['lemmas'])
                    results.append(item)

            results.sort(key=lambda x: x['score'], reverse=True)

            if results:
                st.success(f"🎉 找到 {len(results)} 篇相关真题")
                for res in results[:5]:
                    with st.container(border=True):
                        st.markdown(f"### 【{res['category']}】{res['title']}")
                        st.caption(f"相关度: {round(res['score'] * 100, 1)}% | 命中: {', '.join(res['matches'])}")

                        display_content = res['content']
                        for match_word in res['matches']:
                            pattern = re.compile(r'\b({})\b'.format(re.escape(match_word)), re.IGNORECASE)
                            display_content = pattern.sub(r'<span class="highlight-marker">\1</span>', display_content)
                        st.markdown(display_content, unsafe_allow_html=True)

                        with st.expander("🇨🇳 查看翻译"):
                            st.write(translate_text(res['content']))
            else:
                st.info("未找到匹配文章")
        except ValueError:
            st.warning("输入词汇过于生僻或被停用词过滤")

# --- TAB 2: SmartBridge 生词导入功能 ---
with tab2:
    st.markdown("#### 🔗 SmartBridge：从背单词 App 一键导入")
    st.info(
        "💡 操作指南：在墨墨/不背单词中点击【复制今日单词】，然后直接粘贴在下方。系统将为你推荐包含这些生词最多的真题文章。")

    paste_text = st.text_area("请粘贴生词列表：", height=150, placeholder="例如：\nabandon v. 放弃\nability n. 能力\n...")

    if st.button("📊 生成阅读推荐计划", type="primary", key="btn_bridge"):
        if not paste_text.strip():
            st.warning("⚠️ 请先粘贴内容！")
        elif not filtered_articles:
            st.error("❌ 文章库为空")
        else:
            # 1. 解析粘贴板
            vocab_list = parse_vocabulary_paste(paste_text)

            if not vocab_list:
                st.error("❌ 未识别到有效单词，请检查复制格式。")
            else:
                st.success(f"✅ 成功识别 {len(vocab_list)} 个生词")
                # 展示识别到的词泡泡
                vocab_html = "".join([f'<span class="vocab-badge">{w}</span>' for w in vocab_list])
                st.markdown(vocab_html, unsafe_allow_html=True)

                st.divider()
                st.markdown("### 🏆 今日阅读推荐 (最大覆盖匹配)")

                # 2. 运行最大覆盖算法 (Maximum Coverage)
                # 这是一个简单的统计学逻辑：计算文章中包含了多少个用户生词
                recommendations = []
                user_vocab_set = set(vocab_list)

                for item in filtered_articles:
                    # 计算交集
                    intersection = user_vocab_set.intersection(item['lemmas'])
                    if intersection:
                        recommendations.append({
                            "article": item,
                            "hits": len(intersection),
                            "hit_words": intersection,
                            "coverage": len(intersection) / len(user_vocab_set)
                        })

                # 按命中词数降序排列
                recommendations.sort(key=lambda x: x['hits'], reverse=True)

                if not recommendations:
                    st.warning("🤔 你的生词太生僻了，当前真题库里居然一篇都没碰上...")
                else:
                    for idx, rec in enumerate(recommendations[:5]):
                        art = rec['article']
                        hits = rec['hits']
                        hit_words = rec['hit_words']

                        with st.container(border=True):
                            c1, c2 = st.columns([4, 1])
                            with c1:
                                st.markdown(f"#### Rank {idx + 1} | {art['title']}")
                                st.write(f"包含你生词本中的 **{hits}** 个词")
                            with c2:
                                st.metric("覆盖率", f"{round(rec['coverage'] * 100, 1)}%")

                            # 高亮显示命中的生词
                            display_text = art['content']
                            for hw in hit_words:
                                pattern = re.compile(r'\b({})\b'.format(re.escape(hw)), re.IGNORECASE)
                                display_text = pattern.sub(r'<span class="highlight-marker">\1</span>', display_text)

                            with st.expander("📄 阅读文章 (已高亮生词)"):
                                st.markdown(display_text, unsafe_allow_html=True)
                                st.write("---")
                                st.caption(f"🎯 命中的生词: {', '.join(hit_words)}")