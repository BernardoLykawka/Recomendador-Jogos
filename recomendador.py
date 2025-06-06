import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import re

df = pd.read_csv("games_march_2025_cleaned.csv")

# df.drop(columns=["required_age","dlc_count","reviews","website","support_url",
#                "support_email","windows","mac","linux","metacritic_score",
#                 "metacritic_url","achievements","recommendations","notes","supported_languages",
#                 "full_audio_languages","packages","developers","publishers","screenshots","movies",
#                 "user_score", "score_rank", "average_playtime_2weeks","median_playtime_2weeks","discount",
#                 "peak_ccu","pct_pos_total","pct_pos_recent"], inplace=True)


padrao_excluir = r"(?i)\bhentai\b|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]"      #isso exclui MUITOS jogos de hentai ou com caracteres asiáticos
df = df[~df["name"].str.contains(padrao_excluir, na=False)]          
df = df[df["num_reviews_total"] >= 1000]
df.reset_index(drop=True, inplace=True)


df["positive_pct"] = (df["positive"] / (df["positive"] + df["negative"]) * 100).round(2)
# criei um novo csv com as colunas que eu quero 

# verifica se há colunas com valores nulos - como poucos jogos não possuem essas informações, podemos deixar assim mesmo
# contagem_nulos = df.isnull().sum()
# print(contagem_nulos)

# criei a coluna de porcentagem de avaliações positivas para fazermos um filtro para que apenas alguns jogos sejam recomendados

df_recomendados = df[(df["positive_pct"] >= 73) & (df["num_reviews_total"] > 800) & (df["price"] > 0)]

# pd.create_csv = df.to_csv("games_march_2025_cleaned.csv", index=False)

# print(df_recomendados.shape)
print(df.shape)

# aqui eu combinei o texto para vetorizar
df["text"] = (df["genres"] + " " + 
              df["tags"]+ " " + df["about_the_game"].fillna(""))

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_text = vectorizer.fit_transform(df["text"])
X_combined = X_text.toarray()

def recomendar_jogos(favoritos : list):
    indices_favoritos = df[df["name"].isin(favoritos)].index
    vetor_favorito = X_combined[indices_favoritos].mean(axis=0).reshape(1, -1)

    # calcular similaridade com todos os jogos
    similaridades = cosine_similarity(vetor_favorito, X_combined)[0]

    # ordenar e recomendar
    df["similaridade"] = similaridades
    recomendados = df.loc[df.index.isin(df_recomendados.index)]
    recomendados = recomendados[~recomendados.index.isin(indices_favoritos)]
    recomendados = recomendados.sort_values("similaridade", ascending=False).head(10)
    return recomendados

def print_game(game_row):
    game = game_row[1]
    st.image(game["header_image"])
    st.subheader(game["name"])
    st.write(f"**Gêneros:** {game['genres']}")
    st.write(f"**Categorias:** {game['categories']}")
    st.write(f"**Tags:** {game['tags']}")
    st.write(f"**Sobre o Jogo:** {game['about_the_game']}")


st.set_page_config(page_title="Recomendador de Jogos", page_icon=":game_die:", layout="wide")
st.title("Recomendador de Jogos")
st.write("Selecione seus jogos favoritos para receber recomendações personalizadas.")
favoritos = st.multiselect("Selecione seus jogos favoritos:", df["name"].tolist())
if st.button("Recomendar Jogos"):
    if favoritos:
        st.subheader("Jogos Recomendados")
        recomendados = recomendar_jogos(favoritos)
        for game_row in recomendados.head(10).iterrows():
            print_game(game_row)
    else:
        st.warning("Por favor, selecione pelo menos um jogo favorito.")