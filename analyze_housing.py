import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import matplotlib.image as mpimg
import os


# --- 設定與資料載入 ---

# 創建一個目錄來保存圖表
plots_dir = r'c:\Users\aliso\Downloads\code\code\project\20250721_Vibe_coding\PVC_FinalPresentation\plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# 設定 Matplotlib 的字體以支援中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 載入資料
try:
    housing_df = pd.read_csv(r"c:\Users\aliso\Downloads\code\code\project\20250721_Vibe_coding\PVC_FinalPresentation\housing.csv")
except FileNotFoundError:
    print("錯誤：找不到 housing.csv 檔案。請確認檔案路徑是否正確。")
    exit()

# --- 資料前處理 ---

# 填補 'total_bedrooms' 的缺失值
median_bedrooms = housing_df['total_bedrooms'].median()
housing_df['total_bedrooms'].fillna(median_bedrooms, inplace=True)

print("資料概覽:")
housing_df.info()
print("\n數值資料統計摘要:")
print(housing_df.describe())

# --- 視覺化分析 ---

# 1. 各數值欄位分佈直方圖
print("\n正在產生各欄位分佈直方圖...")
housing_df.hist(bins=50, figsize=(14,10))
plt.suptitle('各數值欄位分佈直方圖', y=0.92, fontsize=16)
# plt.show()
plt.savefig(os.path.join(plots_dir, 'histograms.png'))
plt.close()

# 2. 'ocean_proximity' 類別計數圖
print("正在產生海洋鄰近度計數圖...")
plt.figure(figsize=(10, 6))
sns.countplot(x='ocean_proximity', data=housing_df, order=housing_df['ocean_proximity'].value_counts().index)
plt.title('房屋數量與海洋鄰近度的關係')
plt.xlabel('海洋鄰近度')
plt.ylabel('房屋數量')
plt.savefig(os.path.join(plots_dir, 'ocean_proximity_count.png'))
plt.close()

# 3. 地理位置與房價分佈圖 (加入地圖底圖)
print("正在產生疊加地圖底圖的房價地理分佈圖...")
# 嘗試載入地圖圖片
try:
    california_img = mpimg.imread('california.png')
    # 繪製地圖
    plt.figure(figsize=(12, 10))
    # 設定地圖座標範圍，這個範圍需要與你的地圖圖片匹配
    # 這個範圍適用於常見的 california.png 圖片
    plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5, cmap=plt.get_cmap("jet"))
    
    # 疊加房價散佈圖
    scatter = plt.scatter(housing_df['longitude'], housing_df['latitude'], alpha=0.4,
                          s=housing_df['population']/100, label='人口',
                          c=housing_df['median_house_value'], cmap=plt.get_cmap('jet'))
    
    plt.colorbar(scatter, label='房屋價值中位數')
    plt.xlabel('經度')
    plt.ylabel('緯度')
    plt.title('加州房價地理分佈圖 (點大小代表人數)')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'price_geoplot_with_map.png'))
    plt.close()
except FileNotFoundError:
    print("警告：找不到 'california.png' 地圖底圖檔案，將產生沒有底圖的版本。")
    # 如果找不到地圖，就執行原始的程式碼
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(housing_df['longitude'], housing_df['latitude'], alpha=0.4,
                          s=housing_df['population']/100, label='人口',
                          c=housing_df['median_house_value'], cmap=plt.get_cmap('jet'))
    plt.colorbar(scatter, label='房屋價值中位數')
    plt.xlabel('經度')
    plt.ylabel('緯度')
    plt.title('加州房價地理分佈圖 (點大小代表人數)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'price_geoplot.png'))
    plt.close()

# 3b. 互動式地理位置與房價分佈圖 (使用 Plotly)
print("正在產生互動式房價地理分佈圖 (HTML)...")
fig = px.scatter_mapbox(housing_df,
                        lat="latitude",
                        lon="longitude",
                        color="median_house_value",
                        size="population",
                        hover_name="ocean_proximity",
                        hover_data={"median_house_value": ":,.0f", "population": ":,.0f", "median_income": True},
                        color_continuous_scale=px.colors.cyclical.IceFire,
                        size_max=15,
                        zoom=5,
                        mapbox_style="open-street-map",
                        title="加州房價互動式地理分佈圖",
                        labels={"median_house_value": "房價中位數", "population": "人口"})

fig.update_layout(
    mapbox_center_lon=-119.4179,  # California center
    mapbox_center_lat=36.7783,
    margin={"r": 0, "t": 40, "l": 0, "b": 0},
    title_x=0.5
)

fig.write_html(os.path.join(plots_dir, "interactive_housing_map.html"))

# 4. 數值欄位相關性熱力圖
print("正在產生相關性熱力圖...")
# 選擇數值型欄位進行相關性分析
numeric_df = housing_df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('數值欄位相關性熱力圖')
plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
plt.close()

# 5. 海洋鄰近度與房價的關係 (箱型圖)
print("正在產生海洋鄰近度與房價關係的箱型圖...")
plt.figure(figsize=(10, 6))
sns.boxplot(x='ocean_proximity', y='median_house_value', data=housing_df, 
            order=['NEAR BAY', '<1H OCEAN', 'NEAR OCEAN', 'INLAND', 'ISLAND'])
plt.title('不同海洋鄰近度的房價分佈')
plt.xlabel('海洋鄰近度')
plt.ylabel('房屋價值中位數')
plt.savefig(os.path.join(plots_dir, 'price_by_proximity.png'))
plt.close()

# 6. 收入中位數與房價的關係 (散佈圖)
print("正在產生收入與房價關係的散佈圖...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='median_income', y='median_house_value', data=housing_df, alpha=0.2)
plt.title('收入中位數與房屋價值中位數的關係')
plt.xlabel('收入中位數 (萬美元)')
plt.ylabel('房屋價值中位數')
plt.savefig(os.path.join(plots_dir, 'income_vs_value_scatter.png'))
plt.close()

print("\n所有圖表已成功生成並儲存於 'plots' 資料夾中。")