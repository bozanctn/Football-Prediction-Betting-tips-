import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk

# Veri setini yükleme
data = pd.read_csv("England\England.csv")

data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Toplam gol sayısını hesapla
data["FTTG"] = data["FTHG"] + data["FTAG"]

data = data.fillna(1)

# Takım isimlerini sayılara dönüştürme
team_name_to_id = {
    "Arsenal": 0, "Aston Villa": 1, "Brighton": 2, "Burnley": 3, "Chelsea": 4,
    "Crystal Palace": 5, "Fulham": 6, "Everton": 7, "Leeds": 8, "Leicester": 9,
    "Liverpool": 10, "Man City": 11, "Man United": 12, "Newcastle": 13, "Sheffield": 14,
    "Southampton": 15, "Tottenham": 16, "West Brom": 17, "West Ham": 18, "Wolves": 19,
    "Norwich": 20, "Watford": 21, "Nothingham": 22, "Bournemouth":23, "Brentford": 24, "Luton": 25, "Sheffield United": 26, "Sheffield United": 27, "Ipswich": 28, "Nott'm Forest": 29    # Add all relevant team mappings here
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

# Yeni özellikleri eklemek için bir fonksiyon
def add_last_n_matches_features(data, n=5):
    data['HomeTeam_avg_goals_scored'] = 0.0
    data['HomeTeam_avg_goals_conceded'] = 0.0
    data['AwayTeam_avg_goals_scored'] = 0.0
    data['AwayTeam_avg_goals_conceded'] = 0.0
    data['HomeTeam_avg_corners'] = 0.0
    data['AwayTeam_avg_corners'] = 0.0
    data['HomeTeam_avg_free_kicks'] = 0.0
    data['AwayTeam_avg_free_kicks'] = 0.0
    data['HomeTeam_avg_red_cards'] = 0.0
    data['AwayTeam_avg_red_cards'] = 0.0
    data['HomeTeam_avg_shots_on_target'] = 0.0
    data['AwayTeam_avg_shots_on_target'] = 0.0
    data['HeadToHeadGoals'] = 0.0
    
    teams = data['HomeTeam'].unique()
    
    for team in teams:
        stats = get_last_n_matches_stats(data, team, n)
        data.loc[data['HomeTeam'] == team, 'HomeTeam_avg_goals_scored'] = stats['avg_goals_scored']
        data.loc[data['HomeTeam'] == team, 'HomeTeam_avg_goals_conceded'] = stats['avg_goals_conceded']
        data.loc[data['AwayTeam'] == team, 'AwayTeam_avg_goals_scored'] = stats['avg_goals_scored']
        data.loc[data['AwayTeam'] == team, 'AwayTeam_avg_goals_conceded'] = stats['avg_goals_conceded']
        data.loc[data['HomeTeam'] == team, 'HomeTeam_avg_corners'] = stats['avg_corners']
        data.loc[data['AwayTeam'] == team, 'AwayTeam_avg_corners'] = stats['avg_corners']
        data.loc[data['HomeTeam'] == team, 'HomeTeam_avg_free_kicks'] = stats['avg_free_kicks']
        data.loc[data['AwayTeam'] == team, 'AwayTeam_avg_free_kicks'] = stats['avg_free_kicks']
        data.loc[data['HomeTeam'] == team, 'HomeTeam_avg_red_cards'] = stats['avg_red_cards']
        data.loc[data['AwayTeam'] == team, 'AwayTeam_avg_red_cards'] = stats['avg_red_cards']
        data.loc[data['HomeTeam'] == team, 'HomeTeam_avg_shots_on_target'] = stats['avg_shots_on_target']
        data.loc[data['AwayTeam'] == team, 'AwayTeam_avg_shots_on_target'] = stats['avg_shots_on_target']
    
    for index, row in data.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        data.at[index, 'HeadToHeadGoals'] = get_head_to_head_goals(data, home_team, away_team)
    
    return data

# Yeni özellikleri eklemeye yarayan fonksiyonun kullanımı.
data = add_last_n_matches_features(data)

# Özellikleri ve hedef değişkeni ayarlama
features = ['HomeTeam', 'AwayTeam', 'HomeTeam_avg_goals_scored', 'HomeTeam_avg_goals_conceded', 
            'AwayTeam_avg_goals_scored', 'AwayTeam_avg_goals_conceded', 'HomeTeam_avg_corners', 
            'AwayTeam_avg_corners', 'HomeTeam_avg_free_kicks', 'AwayTeam_avg_free_kicks', 
            'HomeTeam_avg_red_cards', 'AwayTeam_avg_red_cards', 'HomeTeam_avg_shots_on_target', 
            'AwayTeam_avg_shots_on_target', 'HeadToHeadGoals', 'B365H', 'B365D', 'B365A']  # Bahis oranları eklendi

X = data[features]
y = data['FTTG'].values

# Eğitim ve test setlerini oluşturma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturma ve eğitme
model = RandomForestRegressor(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)

# Modelin performansını değerlendirme
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_rmse = root_mean_squared_error(y_train, train_preds)
test_rmse = root_mean_squared_error(y_test, test_preds)


# Tkinter arayüzü 
def predict_match():
    home_team_name = home_team_var.get()
    away_team_name = away_team_var.get()
    
    if home_team_name in team_name_to_id and away_team_name in team_name_to_id:
        home_team = team_name_to_id[home_team_name]
        away_team = team_name_to_id[away_team_name]
        
        home_stats = get_last_n_matches_stats(data, home_team, n=5)
        away_stats = get_last_n_matches_stats(data, away_team, n=5)
        head_to_head_goals = get_head_to_head_goals(data, home_team, away_team)
        
        match_features = [
            home_team, away_team, 
            home_stats['avg_goals_scored'], home_stats['avg_goals_conceded'], 
            away_stats['avg_goals_scored'], away_stats['avg_goals_conceded'], 
            home_stats['avg_corners'], away_stats['avg_corners'], 
            home_stats['avg_free_kicks'], away_stats['avg_free_kicks'], 
            home_stats['avg_red_cards'], away_stats['avg_red_cards'], 
            home_stats['avg_shots_on_target'], away_stats['avg_shots_on_target'], 
            head_to_head_goals, 1.5, 3.5, 6.0  # Bahis oranları (örnek değerler)
        ]
        
        match_features_df = pd.DataFrame([match_features], columns=features)
        predicted_goals = model.predict(match_features_df)[0]
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
