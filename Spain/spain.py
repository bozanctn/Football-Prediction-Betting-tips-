import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk

# Veri seti 
data = pd.read_csv("Spain\Spain.csv")

data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Toplam gol sayısı
data["FTTG"] = data["FTHG"] + data["FTAG"]

data = data.fillna(1)

# Takım isimlerini sayılara dönüştürme
team_name_to_id = {
    "Alaves": 0, "Ath Bilbao": 1, "Ath Madrid": 2, "Barcelona": 3, "Cadiz": 4,
    "Celta": 5, "Eibar": 6, "Elche": 7, "Getafe": 8, "Granada": 9,
    "Huesca": 10, "Levante": 11, "Osasuna": 12, "Betis": 13, "Real Madrid": 14,
    "Sociedad": 15, "Sevilla": 16, "Valencia": 17, "Valladolid": 18, "Villarreal": 19,
    "Espanyol": 20, "Vallecano": 21, "Mallorca": 22, "Almeria": 23, "Girona": 24, "Las Palmas": 25, "Leganes": 26
}

team_id_to_name = {v: k for k, v in team_name_to_id.items()}

data["HomeTeam"] = data["HomeTeam"].map(team_name_to_id)
data["AwayTeam"] = data["AwayTeam"].map(team_name_to_id)

# Her takım için son 5 maçtaki ortalama istatistikleri hesaplayan fonksiyonlar
def get_last_n_matches_stats(data, team, n=5):
    team_matches = data[(data['HomeTeam'] == team) | (data['AwayTeam'] == team)]
    team_matches_sorted = team_matches.sort_values(by='Date', ascending=False).head(n)
    
    stats = {
        'avg_goals_scored': 0,
        'avg_goals_conceded': 0,
        'avg_corners': 0,
        'avg_free_kicks': 0,
        'avg_red_cards': 0,
        'avg_shots_on_target': 0
    }
    
    for _, row in team_matches_sorted.iterrows():
        if row['HomeTeam'] == team:
            stats['avg_goals_scored'] += row['FTHG']
            stats['avg_goals_conceded'] += row['FTAG']
            stats['avg_corners'] += row['HC']
            stats['avg_free_kicks'] += row['HF']
            stats['avg_red_cards'] += row['HR']
            stats['avg_shots_on_target'] += row['HST']
        else:
            stats['avg_goals_scored'] += row['FTAG']
            stats['avg_goals_conceded'] += row['FTHG']
            stats['avg_corners'] += row['AC']
            stats['avg_free_kicks'] += row['AF']
            stats['avg_red_cards'] += row['AR']
            stats['avg_shots_on_target'] += row['AST']
    
    for key in stats:
        stats[key] /= n
    
    return stats

# İki takım arasındaki geçmiş maçlarda toplam gol sayısını hesaplayan fonksiyon
def get_head_to_head_goals(data, home_team, away_team):
    head_to_head_matches = data[((data['HomeTeam'] == home_team) & (data['AwayTeam'] == away_team)) |
                                ((data['HomeTeam'] == away_team) & (data['AwayTeam'] == home_team))]
    total_goals = head_to_head_matches['FTTG'].sum()
    return total_goals

# Yeni özellikleri ekleme
def add_last_n_matches_features(data, n=5):
    data['HeadToHeadGoals'] = 0
    
    for index, row in data.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        data.at[index, 'HeadToHeadGoals'] = get_head_to_head_goals(data, home_team, away_team)
    
    return data

# Yeni özellikleri ekleme fonksiyonunun kullanılması 
data = add_last_n_matches_features(data)

# Özellikleri ve hedef değişkeni ayarlama
features = ['HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']  # Bahis oranları eklendi

X = data[features]
y = data['FTTG'].values

# Eğitim ve test setlerini oluşturma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturma ve eğitme
model = RandomForestRegressor(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)

# Modelin performansını değerlendirme(performansı görmek için sonucu yazıdracak değişikliklerde bulunmalısınız)
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))


# Tkinter arayüzü
def predict_match():
    home_team_name = home_team_var.get()
    away_team_name = away_team_var.get()
    
    if home_team_name in team_name_to_id and away_team_name in team_name_to_id:
        home_team = team_name_to_id[home_team_name]
        away_team = team_name_to_id[away_team_name]
        
        # Özellikleri hesapla
        match_features = pd.DataFrame([{
            'HomeTeam': home_team, 
            'AwayTeam': away_team, 
            'B365H': 1.5,   # Example odds
            'B365D': 3.5,
            'B365A': 6.0
        }])
        
        predicted_goals = model.predict(match_features)[0]
        std_deviation = np.std(test_preds)
        confidence_interval = (predicted_goals - 1.96 * std_deviation, predicted_goals + 1.96 * std_deviation)
        
        result_var.set(f"Tahmini Gol Sayısı: {predicted_goals:.2f}\nGüven Aralığı: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})")
    else:
        result_var.set("Geçersiz takım adı girdiniz.")

app = tk.Tk()
app.title("Maç Tahmin Sistemi")

ttk.Label(app, text="Ev Sahibi Takım:").grid(column=0, row=0, padx=10, pady=10)
ttk.Label(app, text="Deplasman Takımı:").grid(column=0, row=1, padx=10, pady=10)

home_team_var = tk.StringVar()
away_team_var = tk.StringVar()

home_team_entry = ttk.Entry(app, textvariable=home_team_var)
away_team_entry = ttk.Entry(app, textvariable=away_team_var)

home_team_entry.grid(column=1, row=0, padx=10, pady=10)
away_team_entry.grid(column=1, row=1, padx=10, pady=10)

predict_button = ttk.Button(app, text="Tahmin Et", command=predict_match)
predict_button.grid(column=0, row=2, columnspan=2, pady=10)

result_var = tk.StringVar()
result_label = ttk.Label(app, textvariable=result_var)
result_label.grid(column=0, row=3, columnspan=2, pady=10)

app.mainloop()
