import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Logistic_Regression import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def prepare_data_logistic():
    df = pd.read_csv('Part_2_DataCollection/Chuot_PhongVu.csv')

    # Trích xuất đặc trưng từ tên chuột
    df['Is_Gaming'] = df['Name'].str.contains('gaming', case=False, regex=True).astype(np.float32)
    df['Is_Wireless'] = df['Name'].str.contains(r'không dây|wireless|bluetooth|2\.4g', case=False, regex=True).astype(np.float32)
    df['Is_Silent'] = df['Name'].str.contains('silent|chống ồn|yên tĩnh', case=False, regex=True).astype(np.float32)

    # Giữ lại top 10 hãng phổ biến nhất, các hãng còn lại thì là 'Other'
    top_brands = df['Brand'].value_counts().nlargest(10).index
    df['Brand_Grouped'] = df['Brand'].apply(lambda x: x if x in top_brands else 'Other')

    # One-Hot Encoding
    brand_dummies = pd.get_dummies(df['Brand_Grouped'], prefix='Brand', drop_first=True).astype(np.float32)

    # Dùng Scaler chuẩn hóa
    scaler = StandardScaler()
    retail_price_scaled = scaler.fit_transform(df[['Retail_Price']]).astype(np.float32)
    df_price_scaled = pd.DataFrame(retail_price_scaled, columns=['Retail_Price_Scaled'], index=df.index)

    # Kết nối lại tất cả đặc trưng và đưa vào ma trận X
    X_df = pd.concat([df_price_scaled, df[['Is_Gaming', 'Is_Wireless', 'Is_Silent']], brand_dummies], axis=1)
    return df, X_df, scaler

df, X_df, scaler = prepare_data_logistic()
y = df['Is_Gaming'].values.reshape(-1, 1).astype(np.float32)

X = X_df.drop(columns=['Is_Gaming']).values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(learning_rate=0.1, epochs=2000)
model.fit(X_train, y_train)

metrics = model.evaluate(X_test, y_test)

print('Kết quả đánh giá: ')
for metric_name, score in metrics.items():
    print(f'{metric_name}: {score}')

# Trực quan hóa Loss Curve
plt.figure(figsize=(9,5))
plt.plot(range(1, len(model.losses) + 1), model.losses, linewidth=2, label='Binary Cross-Entropy Loss')

plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss-Value')
plt.legend()
plt.show()