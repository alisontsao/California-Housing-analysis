import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


def train_and_evaluate():
    """
    載入資料、進行特徵工程、訓練模型並評估。
    """
    # --- 1. 載入與分割資料 ---
    # 創建一個目錄來保存圖表
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # 設定 Matplotlib 的字體以支援中文
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False

    try:
        housing_df = pd.read_csv(r"c:\Users\aliso\Downloads\code\code\project\20250721_Vibe_coding\PVC_FinalPresentation\housing.csv")
    except FileNotFoundError:
        print("錯誤：找不到 housing.csv 檔案。請確認檔案路徑是否正確。")
        return

    # 將資料分為訓練集與測試集
    train_set, test_set = train_test_split(housing_df, test_size=0.2, random_state=42)

    # 分離特徵 (X) 與目標 (y)
    housing = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()

    # --- 2. 特徵工程 ---
    # 建立新的比例特徵
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]
    
    # 檢視housing資料的敘述統計
    print("資料概覽:")
    print(housing.info())
    print("\n數值資料統計摘要:")
    print(housing.describe())
    # housing.describe().to_csv('housing_describe.csv')
    

    # --- 3. 建立資料前處理 Pipeline ---
    
    # 識別數值與類別欄位
    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    # 建立數值型特徵的處理流程
    # 1. 填補缺失值 (使用中位數)
    # 2. 標準化特徵
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    # 建立完整的處理流程，對不同類型的欄位應用不同的轉換器
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    # --- 4. 準備訓練資料 ---
    housing_prepared = full_pipeline.fit_transform(housing)
    np.savetxt('housing_prepared.csv', housing_prepared, delimiter=',')
    # print("資料已準備完成，並儲存至 'housing_prepared.csv'")

    # --- 5. 訓練與評估模型 ---
    
    # a. 線性迴歸 (Linear Regression)
    models = {
        "線性迴歸": LinearRegression(),
        "隨機森林": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(),
        "梯度提升機": GradientBoostingRegressor(random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"正在訓練 {name} 模型...")
        model.fit(housing_prepared, housing_labels)
        results[name] = model

    # --- 5b. 梯度提升機超參數調校 (Hyperparameter Tuning) ---
    print("\n--- 正在為梯度提升機進行超參數調校 (Grid Search) ---")
    print("這可能需要幾分鐘的時間，請稍候...")
    
    # 記錄開始時間
    start_time = time.time()

    # 定義要測試的參數網格
    # 為了在合理時間內完成，我們先測試一組較小的組合
    param_grid = {
        'n_estimators': [100, 300],         # 樹的數量
        'learning_rate': [0.05, 0.1],       # 學習率
        'max_depth': [3, 5]                 # 每棵樹的最大深度
    }

    # 建立 GridSearchCV 物件
    # cv=3 代表 3-fold cross-validation
    # n_jobs=-1 會使用所有可用的 CPU 核心來加速運算
    grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=3,
                               scoring='neg_mean_squared_error', n_jobs=-1)

    grid_search.fit(housing_prepared, housing_labels)

    print(f"\n網格搜尋完成，耗時: {time.time() - start_time:.2f} 秒")
    print(f"找到的最佳參數: {grid_search.best_params_}")
    results["梯度提升機 (調校後)"] = grid_search.best_estimator_

    # --- 6. 在測試集上進行最終評估 ---
    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()

    # 建立測試集的特徵
    X_test["rooms_per_household"] = X_test["total_rooms"] / X_test["households"]
    X_test["bedrooms_per_room"] = X_test["total_bedrooms"] / X_test["total_rooms"]
    X_test["population_per_household"] = X_test["population"] / X_test["households"]

    # 使用已 fit 的 pipeline 來轉換測試集
    X_test_prepared = full_pipeline.transform(X_test)

    performance_metrics = {}
    print("\n--- 模型評估 ---")

    for name, model in results.items():
        final_predictions = model.predict(X_test_prepared)
        rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
        mae = mean_absolute_error(y_test, final_predictions)
        mape = mean_absolute_percentage_error(y_test, final_predictions)
        r2 = r2_score(y_test, final_predictions)
        
        performance_metrics[name] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R-squared': r2}

        print(f"\n{name}模型:")
        print(f"  - RMSE (均方根誤差):      ${rmse:,.2f}")
        print(f"  - MAE (平均絕對誤差):      ${mae:,.2f}")
        print(f"  - MAPE (平均絕對百分比誤差): {mape:.2%}")
        print(f"  - R² (決定係數):           {r2:.4f}")

    # --- 7. 視覺化成果與結論 ---
    results_df = pd.DataFrame(performance_metrics).T.reset_index().rename(columns={'index': 'Model'})

    # 視覺化 RMSE
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Model', y='RMSE', data=results_df.sort_values('RMSE', ascending=True))
    plt.title('各模型均方根誤差 (RMSE) 比較 (越低越好)', fontsize=16)
    plt.ylabel('RMSE (美元)')
    plt.xlabel('模型')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('plots/model_rmse_comparison.png')
    plt.close()
    print("\n已儲存 RMSE 比較圖至 'plots/model_rmse_comparison.png'")

    # 視覺化 R-squared
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Model', y='R-squared', data=results_df.sort_values('R-squared', ascending=False))
    plt.title('各模型 R² (決定係數) 比較 (越高越好)', fontsize=16)
    plt.ylabel('R² 分數')
    plt.xlabel('模型')
    plt.xticks(rotation=15)
    plt.ylim(0, 1) # R-squared 範圍在 0-1 之間
    plt.tight_layout()
    plt.savefig('plots/model_r2_comparison.png')
    plt.close()
    print("已儲存 R-squared 比較圖至 'plots/model_r2_comparison.png'")

    # 視覺化 MAPE
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Model', y='MAPE', data=results_df.sort_values('MAPE', ascending=True))
    plt.title('各模型平均絕對百分比誤差 (MAPE) 比較 (越低越好)', fontsize=16)
    plt.ylabel('MAPE (%)')
    plt.xlabel('模型')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('plots/model_mape_comparison.png')
    plt.close()
    print("已儲存 MAPE 比較圖至 'plots/model_mape_comparison.png'")

    # 找出最佳模型
    best_model_name = results_df.sort_values('RMSE', ascending=True).iloc[0]['Model']

    print("\n--- 結論 ---")
    print(f"{best_model_name}模型的表現優於所有其他模型。")

if __name__ == '__main__':
    train_and_evaluate()
    
