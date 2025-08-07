import os

def create_html_report():
    """
    生成包含數據分析圖表和結論的 HTML 報告。
    """
    # --- 檔案檢查 ---
    # 定義報告中需要的所有圖檔
    required_files = [
        'plots/histograms.png',
        'plots/ocean_proximity_count.png',
        'plots/interactive_housing_map.html',
        'plots/price_by_proximity.png',
        'plots/correlation_heatmap.png',
        'plots/income_vs_value_scatter.png',
        'plots/model_rmse_comparison.png',
        'plots/model_r2_comparison.png',
        'plots/model_mape_comparison.png'
    ]

    # 檢查所有必要的圖檔是否存在
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("錯誤：缺少以下必要的圖檔，無法生成報告。")
        for f in missing_files:
            print(f" - {f}")
        print("\n請先執行 'analyze_housing.py' 和 'train_housing_model.py' 來生成所有圖表。")
        return

    # HTML 模板
    html_template = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>加州房價數據分析與模型預測報告</title>
    <style>
        * {
            box-sizing: border-box;
        }
        body {{
            font-family: 'Microsoft JhengHei', 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            margin: 0 auto;
            max-width: 1000px;
            padding: 20px;
            color: #333;
            background-color: #f9f9f9;
            text-align: center; /* 將所有內容置中 */
        }}
        h1, h2, h3 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .container {{
            background-color: #fff;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        img {{
            max-width: 50%; /* Adjusted for better visual */
            height: auto;
            display: block;
            margin: 20px auto;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }} 
        iframe { 
            width: 1200px;
            height: 600px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        p, li {{
            font-size: 16px;
            color: #555;
        }}
        ul {{
            padding-left: 20px;
        }}
        .conclusion {{
            background-color: #eaf5ff;
            border-left: 5px solid #3498db;
            padding: 15px;
            margin-top: 20px;
        }}
        .analysis-text {{
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>

    <h1>加州房價數據分析與模型預測報告</h1>

    <div class="container">
        <h2>報告簡介</h2>
        <p>本報告旨在對加州房價數據進行深入的探索性分析。我們將透過一系列的視覺化圖表，探討影響房價的各項因素，並特別針對不同地理區域的次市場進行房價統計分析，以揭示其市場特性與趨勢。</p>
    </div>

    <div class="container">
        <h2>1. 數據概覽與分佈</h2>
        <p class="analysis-text">
            首先，我們檢視數據集中各項特徵的整體分佈情況。
        </p>
        <img src="plots/histograms.png" alt="各數值欄位分佈直方圖">
        <h3>分析：</h3>
        <ul>
            <li><b>數據上限：</b>許多特徵（如 `housing_median_age`, `median_house_value`）的分佈在最右側有突然的截斷，這表明數據可能經過人為設限（Capped），例如房價最高只記錄到 50 萬美元。</li>
            <li><b>長尾分佈：</b>`total_rooms`, `total_bedrooms`, `population`, `households` 等特徵呈現明顯的右偏（長尾）分佈，表示大多數地區的這些數值較低，但有少數人口密集的地區數值極高。</li>
            <li><b>收入分佈：</b>`median_income` 的單位似乎不是美元，且分佈也偏右。在進行模型訓練前，對這些偏態分佈的特徵進行轉換（如取對數）可能會有所幫助。</li>
        </ul>
        <hr>
        <img src="plots/ocean_proximity_count.png" alt="海洋鄰近度計數圖">
        <h3>分析：</h3>
        <ul>
            <li>數據集中最大宗的房屋類型是距離海洋小於一小時車程（`&lt;1H OCEAN`），其次是內陸（`INLAND`）地區。</li>
            <li>島嶼（`ISLAND`）上的房屋數量最少，樣本僅有5筆，這可能使其成為一個特殊的市場，其統計結果需要謹慎解讀。</li>
        </ul>
    </div>

    <div class="container">
        <h2>2. 地理分佈與房價</h2>
        <p class="analysis-text">
            房價與地理位置密切相關。透過互動式地圖，我們可以直觀地看到房價在加州各地的分佈情況。點的大小代表人數多寡，顏色深淺代表房價高低。
        </p>
        <iframe src="plots/interactive_housing_map.html"></iframe>
        <h3>分析：</h3>
        <ul>
            <li><b>沿海效應：</b>房價高的區域（紅色和黃色）主要集中在沿海地帶，特別是舊金山灣區（Bay Area）和南加州沿岸（如洛杉磯、聖地牙哥附近）。</li>
            <li><b>內陸差異：</b>內陸地區的房價普遍較低（藍色和綠色）。</li>
            <li><b>人口與房價：</b>人口密度較高（點較大）的城市區域，其房價也相對較高。</li>
        </ul>
    </div>

    <div class="container">
        <h2>3. 區域次市場分析：不同海洋鄰近度的房價統計</h2>
        <p class="analysis-text">
            我們將 `ocean_proximity` 視為劃分不同次市場的依據，並使用箱型圖來比較各個市場的房價分佈。這有助於我們理解不同區域的市場特性。
        </p>
        <img src="plots/price_by_proximity.png" alt="不同海洋鄰近度的房價分佈">
        <h3>各區域市場分析：</h3>
        <ul>
            <li><b>島嶼 (ISLAND):</b> 房價中位數最高（約 38 萬美元），且價格分佈非常集中，顯示此區房產價值不菲且穩定。但如前述，由於樣本數極少，此結論僅供參考。</li>
            <li><b>海灣附近 (NEAR BAY):</b> 房價中位數僅次於島嶼（約 26 萬美元），價格分佈範圍較廣，從 15 萬到 50 萬美元以上都有，顯示市場的多樣性與活力。</li>
            <li><b>近海 (NEAR OCEAN) &amp; &lt;1H OCEAN:</b> 這兩區的房價中位數相近（約 25 萬和 24 萬美元），且均顯著高於內陸地區。值得注意的是，這兩個區域有大量的房產價格觸及 50 萬美元的上限。</li>
            <li><b>內陸 (INLAND):</b> 房價中位數最低（約 12.5 萬美元），價格分佈也最為集中在較低區間，是加州房價最親民的區域。</li>
        </ul>
        <div class="conclusion">
            <p><b>結論：</b>房屋與海洋的距離是影響房價的關鍵因素。越靠近海岸（特別是海灣地區），房價越高。`ocean_proximity` 成功地將加州房市劃分為幾個具有顯著價格差異的次市場。</p>
        </div>
    </div>

    <div class="container">
        <h2>4. 關鍵影響因子分析</h2>
        <p class="analysis-text">
            為了找出影響房價的其他關鍵因素，我們分析了各數值特徵之間的相關性。
        </p>
        <img src="plots/correlation_heatmap.png" alt="數值欄位相關性熱力圖">
        <img src="plots/income_vs_value_scatter.png" alt="收入與房價關係的散佈圖">
        <h3>分析：</h3>
        <ul>
            <li><b>收入是核心：</b>從熱力圖可見，`median_house_value` 與 `median_income` 呈現最強的正相關（相關係數為 0.69）。這在下方的散佈圖中也得到清晰的驗證。</li>
            <li><b>收入與房價的關係：</b>散佈圖顯示，收入中位數越高的地區，房價中位數也越高。同時，圖中再次確認了房價在 50 萬美元的上限。</li>
            <li><b>其他因素：</b>`latitude`（緯度）和房價也有一定的正相關，可能反映了北加州（如灣區）房價較高的趨勢。而房間總數 (`total_rooms`) 等與房價的相關性反而較弱，這提示我們人均指標（如我們在模型訓練中創建的 `rooms_per_household`）可能比總量指標更具解釋力。</li>
        </ul>
    </div>

    <div class="container">
        <h2>5. 模型訓練與評估</h2>
        <p class="analysis-text">
            為了預測房價，我們訓練了多個機器學習模型，並對其性能進行了評估。
            我們使用了線性迴歸、隨機森林、SVR 和梯度提升機等模型，並對梯度提升機進行了超參數調校。
        </p>
        <img src="plots/model_rmse_comparison.png" alt="各模型RMSE比較">
        <h3>分析：</h3>
        <ul>
            <li><b>RMSE (均方根誤差):</b> RMSE 衡量模型預測值與真實值之間的平均誤差。RMSE 越低表示模型預測越準確。</li>
            <li>從圖中可以看出，<b>梯度提升機 (調校後)</b> 模型的 RMSE 最低，表現最佳。</li>
        </ul>
        <hr>
        <img src="plots/model_r2_comparison.png" alt="各模型R²比較">
        <h3>分析：</h3>
        <ul>
            <li><b>R² (決定係數):</b> R² 衡量模型解釋目標變異的程度。R² 越接近 1 表示模型解釋能力越強。</li>
            <li>同樣地，<b>梯度提升機 (調校後)</b> 模型的 R² 最高，表明它能更好地解釋房價的變異。</li>
        </ul>
        <hr>
        <img src="plots/model_mape_comparison.png" alt="各模型MAPE比較">
        <h3>分析：</h3>
        <ul>
            <li><b>MAPE (平均絕對百分比誤差):</b> MAPE 衡量預測誤差的百分比。MAPE 越低表示預測的相對誤差越小。</li>
            <li><b>梯度提升機 (調校後)</b> 在 MAPE 方面也表現出色，進一步證明其預測的準確性和穩定性。</li>
        </ul>
        <div class="conclusion">
            <p><b>結論：</b>經過超參數調校的梯度提升機模型在所有評估指標上均表現最佳，是本次房價預測的最佳模型。</p>
        </div>
    </div>

    <div class="container">
        <h2>6. 總結</h2>
        <div class="conclusion">
            <p>綜合以上分析，我們可以得出以下主要結論：</p>
            <ol>
                <li><b>地理位置為王：</b>鄰近海洋（特別是海灣地區）是房價最重要的驅動因素。沿海地區的房價遠高於內陸地區。</li>
                <li><b>收入是關鍵：</b>居民的收入中位數與房價有著極強的正相關性，是預測房價的核心指標。</li>
                <li><b>市場存在區隔：</b>加州房市可依據 `ocean_proximity` 明顯區分為數個次市場，各市場的房價水平與分佈特徵差異顯著。</li>
                <li><b>數據限制：</b>數據中房價中位數存在 50 萬美元的上限，這可能會影響模型對高價區的預測準確性，在解讀分析結果與模型預測時需將此納入考量。</li>
                <li><b>最佳模型：</b>梯度提升機模型在經過超參數調校後，展現了最佳的預測性能，可用於未來房價的預測。</li>
            </ol>
        </div>
    </div>

</body>
</html>
    """
    # 產生 HTML 報告檔案
    report_filename = "housing_analysis_report.html"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"報告已成功生成：'{report_filename}'")

if __name__ == '__main__':
    print("正在生成 HTML 分析報告...")
    create_html_report()
    print("報告生成完畢。")
