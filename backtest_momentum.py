import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
import os

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 避免負號變方塊
import matplotlib.pyplot as plt

class MomentumBacktester:
    def __init__(self, db_name="taiwan_momentum_pr_backtest.db"):
        self.db_path = db_name
        self.top_n = 10           # 每次持有檔數
        self.rebalance_days = 21  # 換股週期 (約1個月)
        self.window = 63          # 季度交易日數
        
    def connect(self):
        return sqlite3.connect(self.db_path)

    def load_close_prices(self) -> pd.DataFrame:
        """從資料庫讀取所有股票的收盤價並轉為 Pivot Table"""
        print("讀取本地資料庫股價...")
        with self.connect() as conn:
            df = pd.read_sql_query("""
                SELECT ticker, date, close 
                FROM daily_prices 
                ORDER BY date
            """, conn)
            
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        pivot = df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
        # 填補缺失值 (股票停牌時沿用前一日價格)
        return pivot.ffill()

    def calculate_historical_scores(self, pivot: pd.DataFrame) -> pd.DataFrame:
        """向量化計算歷史上每一天的加權 PR 分數"""
        print("計算歷史動能 PR 分數矩陣...")
        
        # 計算四個季度的滾動報酬率
        ret_q1 = pivot / pivot.shift(self.window) - 1
        ret_q2 = pivot.shift(self.window) / pivot.shift(self.window * 2) - 1
        ret_q3 = pivot.shift(self.window * 2) / pivot.shift(self.window * 3) - 1
        ret_q4 = pivot.shift(self.window * 3) / pivot.shift(self.window * 4) - 1

        # 計算每個季度的橫向 PR 排名 (0 ~ 100)
        pr_q1 = ret_q1.rank(axis=1, pct=True) * 100
        pr_q2 = ret_q2.rank(axis=1, pct=True) * 100
        pr_q3 = ret_q3.rank(axis=1, pct=True) * 100
        pr_q4 = ret_q4.rank(axis=1, pct=True) * 100

        # 依照原策略權重加權
        score = (pr_q1 * 0.40) + (pr_q2 * 0.20) + (pr_q3 * 0.20) + (pr_q4 * 0.20)
        return score

    def fetch_benchmark(self, start_date: str, end_date: str) -> pd.Series:
        """抓取大盤作為比較基準"""
        print("下載加權指數 (^TWII) 作為基準...")
        twii = yf.download("^TWII", start=start_date, end=end_date, progress=False)
        if twii.empty:
            return pd.Series(dtype=float)
        # 確保回傳 Series
        if isinstance(twii.columns, pd.MultiIndex):
            twii = twii['Close']['^TWII']
        else:
            twii = twii['Close']
        twii.index = twii.index.tz_localize(None)
        return twii

    def run_backtest(self):
        pivot = self.load_close_prices()
        if pivot.empty:
            print("資料庫中沒有股價資料，請先執行原選股程式抓取資料。")
            return

        scores = self.calculate_historical_scores(pivot)
        daily_returns = pivot.pct_change() # 每日報酬率

        # 需要至少 4 個季度 (252天) 才能算出第一個分數
        valid_dates = scores.dropna(how="all").index
        if len(valid_dates) < 2:
            print("資料長度不足 252 天，無法進行回測。")
            return
            
        print("開始模擬投資組合交易...")
        
        portfolio_dates = []
        portfolio_values = [1.0] # 初始資金設定為 1 (即 100%)
        current_holdings = []
        
        # 迴圈模擬每日交易
        days_since_rebalance = 0
        
        for i in range(len(valid_dates) - 1):
            today = valid_dates[i]
            tomorrow = valid_dates[i+1]
            portfolio_dates.append(today)
            
            # 定期換股邏輯 (Rebalancing)
            if days_since_rebalance == 0 or days_since_rebalance >= self.rebalance_days:
                # 取得今天的全市場分數
                today_scores = scores.loc[today].dropna()
                # 選出分數最高的前 N 檔股票
                if len(today_scores) >= self.top_n:
                    current_holdings = today_scores.nlargest(self.top_n).index.tolist()
                days_since_rebalance = 0
            
            # 計算明天的資產報酬 (等權重平均)
            if current_holdings:
                # 取得持股明天的日報酬
                stock_rets = daily_returns.loc[tomorrow, current_holdings]
                # 若有股票停牌 (NaN)，以 0 計算
                port_ret = stock_rets.fillna(0).mean()
            else:
                port_ret = 0.0
                
            # 更新資產淨值
            new_value = portfolio_values[-1] * (1 + port_ret)
            portfolio_values.append(new_value)
            
            days_since_rebalance += 1

        portfolio_dates.append(valid_dates[-1])
        
        # 整理成 DataFrame
        result_df = pd.DataFrame({
            "Date": portfolio_dates,
            "Strategy": portfolio_values
        }).set_index("Date")

        # 加入大盤績效
        twii = self.fetch_benchmark(valid_dates[0].strftime("%Y-%m-%d"), 
                                    valid_dates[-1].strftime("%Y-%m-%d"))
        if not twii.empty:
            # 將大盤起點標準化為 1.0
            twii_norm = twii / twii.iloc[0]
            result_df = result_df.join(twii_norm.rename("TWII_Benchmark"), how="left")
            result_df["TWII_Benchmark"] = result_df["TWII_Benchmark"].ffill()

        self.plot_results(result_df)
        self.calculate_metrics(result_df)

    def calculate_metrics(self, df: pd.DataFrame):
        """計算關鍵績效指標 (MDD, 總報酬)"""
        print("\n========== 回測績效報告 ==========")
        for col in df.columns:
            total_return = (df[col].iloc[-1] - 1) * 100
            
            # 歷史最高點 (Rolling Max)
            roll_max = df[col].cummax()
            # 每日回撤 (Drawdown)
            drawdown = df[col] / roll_max - 1
            max_drawdown = drawdown.min() * 100
            
            print(f"[{col}]")
            print(f"  累積報酬率: {total_return:>6.2f}%")
            print(f"  最大回撤 (MDD): {max_drawdown:>6.2f}%")

    def plot_results(self, df: pd.DataFrame):
        """繪製資金曲線"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(df.index, df["Strategy"], label="Top 10 Momentum PR Strategy", linewidth=2, color="#2563eb")
        if "TWII_Benchmark" in df.columns:
            ax.plot(df.index, df["TWII_Benchmark"], label="TWII Benchmark", linewidth=2, color="#6b7280", alpha=0.8)
            
        ax.set_title("動能選股策略 vs 台灣加權指數 (初始資金=1.0)", fontsize=16)
        ax.set_ylabel("Portfolio Value")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        output_file = "momentum_backtest_result.png"
        fig.savefig(output_file, dpi=150)
        print(f"\n已產出回測圖表: {output_file}")

if __name__ == "__main__":
    if not os.path.exists("taiwan_momentum_pr_backtest.db"):
        print("找不到資料庫！請先執行選股程式建立快取資料庫。")
    else:
        tester = MomentumBacktester()
        tester.run_backtest()
