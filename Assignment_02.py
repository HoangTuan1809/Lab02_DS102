import numpy as np
import pandas as pd
from Softmax_Regression import SoftmaxRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def prepare_data_softmax():
    df = pd.read_csv('Part_2_DataCollection/Chuot_PhongVu.csv')

    # Trích xuất đặc trưng từ tên chuột
    df['Is_Gaming'] = df['Name'].str.contains('gaming', case=False, regex=True).astype(np.float32)
    df['Is_Wireless'] = df['Name'].str.contains(r'không dây|wireless|bluetooth|2\.4g', case=False, regex=True).astype(np.float32)
    df['Is_Silent'] = df['Name'].str.contains('silent|chống ồn|yên tĩnh', case=False, regex=True).astype(np.float32)
    df['Is_RGB'] = df['Name'].str.contains(r'rgb|led', case=False, regex=True).astype(np.float32)
    df['DPI_Mentioned'] = df['Name'].str.contains(r'dpi|\d{4,5}', case=False, regex=True).astype(np.float32)

    # Xử lý cột Discount_Percent
    df['Discount_Ratio'] = (df['Discount_Percent'] / 100.0).astype(np.float32)

    # Scale số tiền giảm giá
    scaler = StandardScaler()
    discount_amount_scaler = scaler.fit_transform(df[['Discount_Amount']]).astype(np.float32)
    df_discount_scaled = pd.DataFrame(discount_amount_scaler, columns=['Discount_Amount_Scaled'], index=df.index)

    # Mã hóa One-Hot Encoding cho top 5 hãng
    top_brands = df['Brand'].value_counts().nlargest(5).index
    df['Brand_Grouped'] = df['Brand'].apply(lambda x: x if x in top_brands else 'Other')
    brand_dummies = pd.get_dummies(df['Brand_Grouped'], prefix='Brand', drop_first=True).astype(np.float32)

    # Gom tất cả vào ma trận X
    X_df = pd.concat([df[['Is_Gaming', 'Is_Wireless', 'Is_Silent', 'Is_RGB', 'Discount_Ratio', 'DPI_Mentioned']], df_discount_scaled, brand_dummies], axis=1)

    # Phân khúc giá thành 5 lớp: 0, 1, 2, 3, 4
    bins = [-1, 250_000, 500_000, 1_000_000, 2_000_000, np.inf]
    y = pd.cut(df['Retail_Price'], bins=bins, labels=[0, 1, 2, 3, 4]).astype(int).values

    return df, X_df, y

df, X_df, y= prepare_data_softmax()
X_train, X_test, y_train, y_test = train_test_split(X_df.values.astype(np.float32), y, test_size=0.2, random_state=42, stratify=y)

model = SoftmaxRegression(learning_rate=0.5, epochs=20000, num_classes=5)
model.fit(X_train, y_train)

target_names = ['Siêu rẻ (<250K)', 'Rẻ (250K-500K)', 'Trung (500K-1M)', 'Khá mắc (1M-2M)', 'Siêu mắc (>2M)']
metrics = model.evaluate(X_test, y_test, target_names=target_names)

print('Kết quả đánh giá: ')
print(metrics['Classification Report'])

# Trực quan hóa Loss Curve
plt.figure(figsize=[9,5])
plt.plot(range(1, len(model.losses) + 1), model.losses, linewidth=2, label='Categorical Cross-Entropy Loss')
plt.title('Loss Curve - 5 Classes')
plt.xlabel('Epochs')
plt.ylabel('Loss-Value')
plt.legend()
plt.show()