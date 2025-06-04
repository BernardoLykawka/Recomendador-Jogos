import pandas as pd

df = pd.read_csv("games_march_2025_cleaned.csv")

# df.drop(columns=["required_age","dlc_count","reviews","website","support_url",
#                "support_email","windows","mac","linux","metacritic_score",
#                 "metacritic_url","achievements","recommendations","notes","supported_languages",
#                 "full_audio_languages","packages","developers","publishers","screenshots","movies",
#                 "user_score", "score_rank", "average_playtime_2weeks","median_playtime_2weeks","discount",
#                 "peak_ccu","pct_pos_total","pct_pos_recent"], inplace=True)

# criei um novo csv com as colunas que eu quero 


# verifica se há colunas com valores nulos - como poucos jogos não possuem essas informações, podemos deixar assim mesmo
contagem_nulos = df.isnull().sum()
print(contagem_nulos)

# criei a coluna de porcentagem de avaliações positivas para fazermos um filtro para que apenas alguns jogos sejam recomendados
# df["positive_pct"] = df["positive"] / (df["positive"] + df["negative"]) * 100 
df_recomendados = df[(df["positive_pct"] >= 73) & (df["num_reviews_total"] > 3000)]

pd.create_csv = df.to_csv("games_march_2025_cleaned.csv", index=False)

print(df_recomendados.shape)
print(df.shape)